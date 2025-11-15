from cgitb import text
import json
import re
import base64
import logging
import os
from io import BytesIO
from typing import Dict, Any, List, Union, Optional, Tuple
from PIL import Image
import copy
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from qwen_vl_utils import smart_resize

from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# from verl.utils.reward_score import zoomin_reward

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))




class ZoominVisionTool(BaseTool):
    """A tool for image zoom-in functionality vision tasks.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": "Zoom in on a specific region of an image by cropping it based on a bounding box",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2]",
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional label for the object being zoomed in",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Configuration parameters
        self.enlarge_factor = config.get('enlarge_factor', 1.5)
        self.scaleup_factor = config.get('scaleup_factor', 1.5)
        self.min_resolution = config.get('min_resolution', 224)
        self.min_pixels = self.min_resolution ** 2 if self.min_resolution != 0 else 4 * 28 * 28
        self.image_data = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    


    # async def create(self, instance_id: Optional[str] = None, question: Optional[list] = None, image_path: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
       
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Extract image path from kwargs
        # if image_path is None:
        image_path = kwargs.get("create_kwargs").get("image_path")
        question = kwargs.get("create_kwargs").get("question")
        ground_truth = kwargs.get("create_kwargs").get("ground_truth")

        from verl.utils.dataset.vision_utils import process_image, process_video

        image = Image.open(image_path)
        image_data = process_image(image)
        self.image_data = image_data

        
        self._instance_dict[instance_id] = {
            "question": question,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "reward": 0.0,
        }

        return instance_id, ToolResponse()

 

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str | Dict[str, Any], float, dict]:
        """Execute the zoom-in tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The reward score of the tool.
            tool_metrics: The metrics of the tool.
        """

        # print("***** parameters: ", parameters) # bbox_2d
        # print("***** instance_id: ", instance_id)
        # print("***** kwargs: ", kwargs) # image_path
        # reward = await self.calc_reward(instance_id)
        

        #     # Convert metadata to metrics
        # metrics = {"query_count": metadata.get("query_count", 0), "status": metadata.get("status", "unknown"), "total_results": metadata.get("total_results", 0), "api_request_error": metadata.get("api_request_error")}

        # Initialize return values
        bbox_2d = parameters.get("bbox_2d", [])
        
        # Get image path from kwargs or instance data
        image_path = kwargs.get("image_path") or self._instance_dict.get(instance_id, {}).get("image_path")
        USER_PROMPT = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
        metrics = {}

        # print("*************************************************************************************")
        # print("***** instance_id: ", self._instance_dict[instance_id])
        # print("***** image_path: ", image_path)
        # print("***** bbox_2d: ", bbox_2d)


        # Store the tool call for later reward calculation
        try:
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["bbox"] = bbox_2d
                self._instance_dict[instance_id]["tool_called"] = True

            # original_image = Image.open(image_path).convert("RGB")
            # resized_image = self.resize_image(original_image, min_pixels=self.min_pixels)
            # resized_image = image_data[0]
            # original_image = image_data[0]
            original_image = self.image_data
            # resized_image = self.resize_image(original_image, min_pixels=self.min_pixels)
            resized_image = original_image

            W, H = resized_image.size
            
            # Validate bbox coordinates
            # x0, y0, x1, y1 = bbox_2d
            # print("***** bbox_2d: ", bbox_2d)
            is_valid, cropped_bbox_msg = self.check_absolute_bbox_format(bbox_2d, W, H)
            print("***** cropped_bbox_msg: ", cropped_bbox_msg)
            
            if is_valid:
                # Create tool call text for do_crop function
                # tool_call_text = f"<tool_call>{{'name': 'image_zoom_in_tool', 'arguments': {{'bbox_2d': {bbox_2d}}}}}</tool_call>"
                
                # Perform cropping
                cropped_image = self.do_crop(
                    resized_image,
                    bbox_2d,
                    # scaleup_factor=self.scaleup_factor,
                    # enlarge_factor=self.enlarge_factor,
                    # min_pixels=self.min_pixels
                )
                
                # Store cropped image
                self._instance_dict[instance_id]["cropped_image"] = cropped_image
                self._instance_dict[instance_id]["original_image"] = resized_image
                

                metrics["status"] = "success"
                # metrics["cropped"] = True
                metrics["bbox_valid"] = True
                metrics["original_size"] = resized_image.size
                metrics["cropped_size"] = cropped_image.size


                
                
                result_text = {
                    "image": [cropped_image],
                    "text": USER_PROMPT
                } 


                print(f"Successfully cropped image with bbox {bbox_2d}. Original: {resized_image.size}, Cropped: {cropped_image.size}")
            else:
                result_text = {
                    "image": [],
                    "text": f"{cropped_bbox_msg}"
                    # "text": USER_PROMPT
                }

                metrics["status"] = "error"
                metrics["error"] = "invalid bbox format"


                 
        except Exception as img_error:
            import traceback
            traceback.print_exc()
                  # 获取完整的traceback字符串
            error_msg = traceback.format_exc()

            print(f"***** [Error] when cropping image: {img_error}")
            print(f"***** Full traceback:\n{error_msg}")

            result_text = {
                "image": [],
                "text": f"Error: {str(img_error)}"
                # "text": USER_PROMPT
            }

            metrics["status"] = "error"

            metrics["error"] = "image_processing_failed"

            
    

        # print(f"***** EXECUTE RESULT: {result_text}")
        # print(f"***** EXECUTE METRICS: {metrics}")
        tool_response = result_text
        tool_reward_score = 0.0
        tool_metrics = metrics

        # return ToolResponse(text=tool_response["text"]), tool_reward_score, tool_metrics

        if len(tool_response["image"]) > 0:
            return ToolResponse(image=tool_response["image"], text=tool_response["text"]), tool_reward_score, tool_metrics
        else:
            return ToolResponse(text=tool_response["text"]), tool_reward_score, tool_metrics




    # async def calc_reward(self, instance_id: str, **kwargs) -> float:
    #     print("**** instance_id: ", instance_id)
    #     print("**** self._instance_dict[instance_id]: ", self._instance_dict[instance_id])
    #     return zoomin_reward.compute_score(
    #         self._instance_dict[instance_id]["response"],
    #         self._instance_dict[instance_id]["ground_truth"],
    #         reward_type="strict",
    #     )
    
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]





    def resize_image(self, original_image, factor=28, min_pixels=4*28*28, max_pixels=5120*28*28):
        """
        Resize the image using smart_resize function to meet the model's requirements.
        
        Args:
            original_image: PIL Image object
            factor: The factor by which dimensions should be divisible (default: 14)
            min_pixels: Minimum total pixels (default: 336*336)
            max_pixels: Maximum total pixels (default: 672*672)
        
        Returns:
            Resized PIL Image
        """
        # print("***** min_pixels:", min_pixels)

        if isinstance(original_image, Image.Image):
            # Get original dimensions
            original_width, original_height = original_image.size
            
            # Calculate new dimensions using smart_resize
            new_height, new_width = smart_resize(
                height=original_height,
                width=original_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            # Resize the image while maintaining aspect ratio
            resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # print("Original image: ", original_image.size)
            # print("Resized image: ", resized_image.size)
        
            return resized_image
        
        elif isinstance(original_image, tuple):
            original_width, original_height = original_image[0], original_image[1]
            new_height, new_width = smart_resize(
                height=original_height,
                width=original_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            return (new_width, new_height)
        

    def bbox_out_of_range(self, bbox, w, h):
        """Generate error message for bounding box coordinates out of image range."""
        x0, y0, x1, y1 = bbox
        error_details = []
        
        if x0 < 0:
            error_details.append(f"x0 ({x0}) < 0")
        if y0 < 0:
            error_details.append(f"y0 ({y0}) < 0")
        if x1 > w:
            error_details.append(f"x1 ({x1}) > image width ({w})")
        if y1 > h:
            error_details.append(f"y1 ({y1}) > image height ({h})")
        
        return f"Invalid bounding box: Bounding box {bbox} coordinates out of image range [0,0,{w},{h}]. Issues: {', '.join(error_details)}"
    
    def bbox_invalid_format(self, bbox):
        """Generate error message for invalid bounding box format."""
        x0, y0, x1, y1 = bbox
        error_details = []
        
        if x1 <= x0:
            error_details.append(f"x1 ({x1}) must be greater than x0 ({x0})")
        if y1 <= y0:
            error_details.append(f"y1 ({y1}) must be greater than y0 ({y0})")
        
        return f"Invalid bounding box: Bounding box {bbox} has invalid coordinates. Issues: {', '.join(error_details)}"


    def check_absolute_bbox_format(self, bbox, w, h):
        x0, y0, x1, y1 = bbox
        if not isinstance(bbox, list):
            print(f"[Error] 🔴 Bounding box {bbox} is not a list")
            text = f"Invalid bounding box: Bounding box must be a list of 4 numbers, but got a {type(bbox)}: '{bbox}'"

            return False, text
        if len(bbox) != 4:
            print(f"[Error] 🔴 Bounding box {bbox} length error, must be 4 elements")
            text = f"Invalid bounding box: Bounding box must be a list of 4 numbers, but got length {len(bbox)}: '{bbox}'"

            return False, text
        if not all(isinstance(i, (int, float)) for i in bbox):
            print(f"[Error] 🔴 Bounding box {bbox} elements must be numbers")
            text = f"Invalid bounding box: Bounding box must be a list of 4 numbers, but found non-numeric elements in: '{bbox}'"
            
            return False, text

        if not (0 <= x0 < w and 0 <= y0 < h and 0 < x1 <= w and 0 < y1 <= h):
            # print(f"[Error] 🔴 Bounding box {bbox} value out of image range [0,0,{w},{h}]")
            # text = f"Bounding box {bbox} value out of image range [0,0,{w},{h}]"
            text = self.bbox_out_of_range(bbox, w, h)
            print(f"[Error] 🔴 {text}")


            return False, text
        if x1 <= x0 or y1 <= y0:
            # print(f"[Error] 🔴 Bounding box {bbox} coordinates are invalid, must be x1 > x0 and y1 > y0")
            # text = f"Bounding box {bbox} coordinates are invalid, must be x1 > x0 and y1 > y0"
            text = self.bbox_invalid_format(bbox)
            print(f"[Error] 🔴 {text}")


            return False, text
        else:
            print(f"[Info] ✅ Bounding box format is correct: {bbox} in image range [0,0,{w},{h}]")
            text = f"Bounding box format is correct: {bbox} in image range [0,0,{w},{h}]"

            return True, text

    def enlarge_bbox(self, bbox_2d, image, enlarge_factor):
        W, H = image.size
        x0, y0, x1, y1 = bbox_2d

        if enlarge_factor != 1:
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            bw = (x1 - x0) * enlarge_factor
            bh = (y1 - y0) * enlarge_factor
            
            x0 = max(0, cx - bw / 2)
            y0 = max(0, cy - bh / 2)
            x1 = min(W, cx + bw / 2)
            y1 = min(H, cy + bh / 2)
        
        x0_int = int(x0)
        y0_int = int(y0)
        x1_int = int(x1)
        y1_int = int(y1)
        
        return [x0_int, y0_int, x1_int, y1_int]

    def do_crop(
        self,
        image: Image.Image, 
        bbox_2d: list,
        # scaleup_factor = 1,
        # enlarge_factor = 1.5,
        # min_pixels = 4*28*28
        ) -> list[Image.Image]:
        # print("***** enlarge_factor: ", enlarge_factor)
        
        # decoded_text = clean_output(decoded_text)
        image_tozoomin = image
        # print("***** image_tozoomin.size: ", image_tozoomin.size)
        

        bbox_zoom_in = self.enlarge_bbox(bbox_2d, image_tozoomin, self.enlarge_factor)
    
        # print("***** bbox_zoom_in_after_enlarge: ", bbox_zoom_in)
        # print("***** scaleup_factor: ", scaleup_factor)
        
        # try:
        image_zoom_in = image_tozoomin.crop(bbox_zoom_in)
        # print("***** image_zoom_in.size after crop: ", image_zoom_in.size)

        resize_img_size = (image_zoom_in.size[0]*self.scaleup_factor, image_zoom_in.size[1]*self.scaleup_factor)
        new_size_width, new_size_height = self.resize_image(resize_img_size, min_pixels=self.min_pixels)
        image_zoom_in = image_zoom_in.resize((new_size_width, new_size_height), Image.Resampling.LANCZOS).convert("RGB").copy()

            # print("***** image_zoom_in.size after scaleup: ", image_zoom_in.size)

        # except Exception as e:
        #     print(f"[**Error**] Error cropping image: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     image_zoom_in = image_tozoomin

        return image_zoom_in



    # def encode_pil_image_to_base64(self, pil_image):
    #     buffered = BytesIO()
    #     pil_image.save(buffered, format="PNG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    #     return img_str
