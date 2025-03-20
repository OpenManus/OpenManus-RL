import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Type


def get_openmanus_path():
    """获取OpenManus目录的绝对路径"""
    path = Path(os.getcwd()).parent / "OpenManus"

    print("using OpenManus path:", path)
    if path.exists() and path.is_dir():
        return str(path.absolute())

    # 如果找不到，尝试一个固定的相对路径
    return "./OpenManus"


# 获取并添加OpenManus路径
openmanus_path = get_openmanus_path()

# 确保OpenManus路径只被添加一次
if openmanus_path not in sys.path:
    sys.path.insert(0, openmanus_path)


# 导入常用的OpenManus工具
def import_tools() -> Dict[str, Type]:
    """导入并返回OpenManus基础工具"""
    try:
        from app.tool import BrowserUseTool, StrReplaceEditor, Terminate, ToolCollection
        from app.tool.python_execute import PythonExecute

        return {
            "browser_use": BrowserUseTool,
            "terminate": Terminate,
            "python_execute": PythonExecute,
            "str_replace_editor": StrReplaceEditor,
        }
    except ImportError as e:
        print(f"导入OpenManus工具时出错: {e}")
        print(f"当前sys.path: {sys.path}")
        print(f"OpenManus路径: {openmanus_path}")
        raise


def get_tool_collection(tool_list: List[str] = None) -> Any:
    """
    获取工具集合

    Args:
        tool_list: 需要的工具列表，默认为["BrowserUseTool", "PythonExecute", "Terminate"]

    Returns:
        ToolCollection实例
    """
    from app.tool import ToolCollection

    if tool_list is None:
        tool_list = ["browser_use", "python_execute", "terminate"]

    tools_dict = import_tools()

    all_tools = ToolCollection(
        *(tools_dict[tool]() for tool in tool_list if tool in tools_dict)
    )
    return all_tools


if __name__ == "__main__":

    def test_str_replace_editor():

        import asyncio
        import os

        test_path = os.path.join(os.getcwd(), "hello.txt")
        f = open(test_path, "w")
        f.write("hello world")
        f.close()
        tools = get_tool_collection(["str_replace_editor"])
        result = asyncio.run(
            tools.execute(
                name="str_replace_editor",
                tool_input={
                    "command": "str_replace",
                    "path": test_path,
                    "old_str": "hello",
                    "new_str": "hi",
                },
            )
        )
        print(result)

    test_str_replace_editor()
