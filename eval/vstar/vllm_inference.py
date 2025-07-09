import argparse
import base64
import copy
import json
import os
import re
from dataclasses import asdict
from io import BytesIO

from PIL import Image
from tqdm import tqdm
from vllm import LLM, EngineArgs, SamplingParams

from qwen_vl_utils import smart_resize


SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an optional object label.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

USER_PROMPT = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."



def resize_image(original_image, factor=28, min_pixels=4*28*28, max_pixels=3840*3840):
    """
    Resize an image or image size tuple to meet pixel count constraints while preserving aspect ratio.

    Parameters:
        original_image (PIL.Image.Image or tuple): The input image (as a PIL Image) or its (width, height) tuple.
        factor (int): The resizing factor. Output dimensions will be multiples of this value.
        min_pixels (int): Minimum allowed pixel count.
        max_pixels (int): Maximum allowed pixel count.

    Returns:
        PIL.Image.Image or tuple: The resized image (if input is PIL Image) or new size tuple (if input is a tuple).
    """
    if isinstance(original_image, Image.Image):
        original_width, original_height = original_image.size
        new_height, new_width = smart_resize(
            height=original_height,
            width=original_width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
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


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str


def load_7b_model(model_path):
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 6},
        disable_mm_preprocessor_cache=True,
    )
    engine_args = asdict(engine_args)
    llm = LLM(**engine_args)

    return llm


def check_absolute_bbox_format(bbox, w, h):
    """
    Check whether a bounding box is in valid absolute format within image dimensions.

    Parameters:
        bbox (list): Bounding box in [x0, y0, x1, y1] format.
        w (int or float): Width of the image.
        h (int or float): Height of the image.

    Returns:
        (bool, str): (True, message) if valid; otherwise (False, error message).
    """
    if not isinstance(bbox, list):
        return False, "[WARNING] Bounding box must be a list."

    if len(bbox) != 4:
        return False, f"[WARNING] Bounding box must contain 4 elements: {bbox}"

    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False, f"[WARNING] All bounding box elements must be numeric: {bbox}"

    x0, y0, x1, y1 = bbox

    if not (0 <= x0 < w and 0 <= y0 < h and 0 < x1 <= w and 0 < y1 <= h):
        return False, f"[WARNING] Bounding box values must be within image bounds [0, 0, {w}, {h}]: {bbox}"

    if x1 <= x0 or y1 <= y0:
        return False, f"[WARNING] Invalid bounding box coordinates: x1 must be > x0 and y1 must be > y0: {bbox}"

    print(f"[INFO] Valid absolute bounding box: {bbox}")

    return True, "Bounding box format is valid."


def find_absolute_bboxes(outputs_string, image, enlarge_factor=1):
    """
    Extract and validate an absolute bounding box from a tool_call string,
    optionally enlarging the box by a given factor.

    Parameters:
        outputs_string (str): The string output containing a <tool_call>...</tool_call> block 
                              with bbox information in JSON format.
        image (PIL.Image): The image associated with the bounding box. Used to validate bounds.
        enlarge_factor (float): Optional factor to enlarge the bounding box while preserving its center.
                                Must be >= 1. Defaults to 1 (no enlargement).

    Returns:
        list[int] or None: A list [x0, y0, x1, y1] of integer bounding box coordinates
                           if parsing and validation succeed; otherwise None.

    Notes:
        - The function assumes that the bbox is in absolute coordinates.
        - If the bbox is invalid or parsing fails, a warning is printed and None is returned.
        - Enlargement is clipped to the image boundaries to avoid overflow.
    """
    W, H = image.size

    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_call_match = re.search(tool_call_pattern, outputs_string, re.DOTALL)
    
    if not tool_call_match:
        print("[WARNING] No tool_call found in outputs_string.")
        return None
    
    tool_call_content = tool_call_match.group(1).strip()
        
    try:
        tool_call_data = json.loads(tool_call_content)
        
        bbox_content = tool_call_data["arguments"]["bbox_2d"]
        x0, y0, x1, y1 = bbox_content
            
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse tool_call JSON: {e}")
        return None
    
    is_valid, error_msg = check_absolute_bbox_format([x0, y0, x1, y1], W, H)

    if not is_valid:
        print(f"[WARNING] Invalid bbox: {[x0, y0, x1, y1]} - {error_msg}")
        return None
    
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
    image: Image.Image, 
    decoded_text: str,
    scaleup_factor: int = 2,
    enlarge_factor: float = 1.5,
    min_pixels: int = 4 * 28 * 28
) -> Image.Image:
    """
    Crop an image region based on a decoded text string containing bbox info, 
    then resize it with optional scaling and enlargement.

    Parameters:
        image (PIL.Image): The original input image.
        decoded_text (str): Model output string containing <tool_call> JSON with bbox coordinates.
        scaleup_factor (int): Factor to scale up the cropped region. Default is 2.
        enlarge_factor (float): Factor to enlarge the bbox before cropping. Default is 1.5.
        min_pixels (int): Minimum pixel count constraint for the resized output. Default is 4*28*28.

    Returns:
        PIL.Image: The cropped and resized image. If an error occurs, the original image is returned.
    """
    image_to_crop = image
    bbox_zoom_in = find_absolute_bboxes(decoded_text, image_to_crop, enlarge_factor)

    try:
        image_zoom_in = image_to_crop.crop(bbox_zoom_in)
        resize_img_size = (
            image_zoom_in.size[0] * scaleup_factor,
            image_zoom_in.size[1] * scaleup_factor
        )
        new_width, new_height = resize_image(resize_img_size, min_pixels=min_pixels)
        image_zoom_in = image_zoom_in.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        ).convert("RGB").copy()

    except Exception as e:
        print(f"[ERROR] Error cropping image: {e}")
        import traceback
        traceback.print_exc()
        image_zoom_in = image_to_crop

    return image_zoom_in


def get_print_message(print_message):
    for msg in print_message:
        content = msg["content"]

        processed_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                processed_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,base64_image"}})
            else:
                processed_content.append(item)
        msg["content"] = processed_content

    return print_message
    
    

def run_inference_loop(llm, test_data, args, min_pixels):
    """
    Generate answers for a list of test data using a given model.

    Parameters:
        llm (vllm.LLM): The model to use for generation.
        test_data (list): A list of dictionaries containing test data.
        args (argparse.Namespace): Command line arguments.
        min_pixels (int): Minimum pixel count constraint for image resizing.

    Returns:
        list: A list of dictionaries containing the results of the generation.
    """

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0
    )
    max_iterations = 6
    results = []
    
    for data_idx, data in tqdm(enumerate(test_data), total=len(test_data), desc="Processing samples"):

        image_name = data['image']
        text = data['text']
        label = data['label']

        image_path = os.path.join(args.vstar_bench_path, image_name)
        
        formatted_question = f"Question: {text}{USER_PROMPT}"
        
        image_original = Image.open(image_path).convert('RGB')
        resized_image = resize_image(image_original, min_pixels=min_pixels)

        base64_image = encode_pil_image_to_base64(resized_image)
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": formatted_question},
                ],
            }
        ]

        chat_message = message
        iteration = 0


        try:
            answer = ""
            while iteration < max_iterations:
                print(f"[INFO] iteration: {iteration}")
                print_message = get_print_message(copy.deepcopy(chat_message))
                print(f"[INFO] print_message\n{print_message}")
                output = llm.chat(chat_message, sampling_params)

                generated_text = output[0].outputs[0].text
                generated_token_ids = output[0].outputs[0].token_ids

                if "<tool_call>" in generated_text:
                    try:
                        image_zoomin = do_crop(
                            resized_image,
                            generated_text,
                            scaleup_factor=args.scaleup_factor,
                            enlarge_factor=args.enlarge_factor,
                            min_pixels=min_pixels
                        )
                                           
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        image_zoomin = resized_image
                    
                    base64_image = encode_pil_image_to_base64(image_zoomin)
                    chat_message.extend([
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": generated_text},
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                {"type": "text", "text": USER_PROMPT},
                            ]
                        }])
                    iteration += 1
                else:
                    answer = generated_text
                    chat_message.extend([
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": generated_text},
                            ]
                        },
                    ])
                    break

            
            status = 'success'  
            save_info = {}
            save_info['image'] = image_path
            save_info['question'] = text
            save_info['answer'] = label
            save_info['pred_ans'] = answer
            save_info['pred_output'] = get_print_message(copy.deepcopy(chat_message))
            save_info['status'] = status
            
            results.append(save_info)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error] Error when processing data {data_idx}: {e}")
            
            save_info = {}
            save_info['image'] = image_path
            save_info['question'] = text
            save_info['answer'] = label
            save_info['pred_ans'] = ""
            save_info['pred_output'] = ""
            save_info['status'] = f"error: {e}"
            
            results.append(save_info)

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enlarge_factor", type=float, default=1.5)
    parser.add_argument("--scaleup_factor", type=float, default=2)
    parser.add_argument("--vstar_bench_path", type=str, default=None, required=True)
    parser.add_argument("--min_resolution", type=int, default=112)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="model")

    args = parser.parse_args()

    min_pixels = args.min_resolution**2 if args.min_resolution != 0 else 4*28*28
    print(f"[INFO] min_pixels: {min_pixels}")

    json_file = f"{args.vstar_bench_path}/test_questions.jsonl"

    direct_attributes_data = []
    relative_position_data = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            category = data.get('category', '')
            
            if category == 'direct_attributes':
                direct_attributes_data.append(data)
            elif category == 'relative_position':
                relative_position_data.append(data)
    
    print(f"Found {len(direct_attributes_data)} direct_attributes samples")
    print(f"Found {len(relative_position_data)} relative_position samples")

    # Load model
    llm = load_7b_model(args.model_path)

    for test_type in ['direct_attributes', 'relative_position'] :
        save_name = f"result_{test_type}_{args.save_name}.jsonl"

        if test_type == 'direct_attributes':
            test_data = direct_attributes_data
        elif test_type == 'relative_position':
            test_data = relative_position_data

        
        results = run_inference_loop(llm, test_data, args, min_pixels)
        
        # Save results
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_file_path = os.path.join(args.save_dir, save_name)
            
            with open(save_file_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"[INFO] Results saved to: {save_file_path}")
        else:
            print("[ERROR] No save path specified, results not saved")
