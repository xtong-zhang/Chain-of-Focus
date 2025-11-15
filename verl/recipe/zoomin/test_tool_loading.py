#!/usr/bin/env python3
"""
测试脚本：验证 ZoominVisionTool 是否能被正确加载
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_tool_loading():
    """测试工具加载"""
    try:
        print("🔍 测试工具加载...")
        
        # 测试导入
        from recipe.zoomin.zoomin_vision_tool import ZoominVisionTool
        print("✅ 成功导入 ZoominVisionTool")
        
        # 测试工具模式
        from verl.tools.schemas import OpenAIFunctionToolSchema
        
        # 创建工具模式
        tool_schema = OpenAIFunctionToolSchema.model_validate({
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
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        
        print("✅ 成功创建工具模式")
        
        # 测试工具实例化
        config = {
            'enlarge_factor': 1.5,
            'scaleup_factor': 2,
            'min_resolution': 112
        }
        
        tool = ZoominVisionTool(config, tool_schema)
        print("✅ 成功创建工具实例")
        print(f"🔧 工具名称: {tool.name}")
        print(f"🔧 工具模式: {tool.tool_schema}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_registry():
    """测试工具注册"""
    try:
        print("\n🔍 测试工具注册...")
        
        from verl.tools.utils.tool_registry import initialize_tools_from_config
        
        # 获取当前配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), "zoomin_tool_config.yaml")
        print(f"📁 配置文件路径: {config_path}")
        
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        # 加载工具
        tools = initialize_tools_from_config(config_path)
        print(f"✅ 成功加载 {len(tools)} 个工具")
        
        for i, tool in enumerate(tools):
            print(f"🔧 工具 {i+1}: {tool.name}")
            print(f"🔧 工具类型: {type(tool)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具注册失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始测试 ZoominVisionTool...")
    print("=" * 60)
    
    # 测试工具加载
    tool_loading_success = test_tool_loading()
    
    # 测试工具注册
    tool_registry_success = test_tool_registry()
    
    print("=" * 60)
    if tool_loading_success and tool_registry_success:
        print("🎉 所有测试通过！工具应该可以正常工作。")
    else:
        print("💥 测试失败，请检查错误信息。") 