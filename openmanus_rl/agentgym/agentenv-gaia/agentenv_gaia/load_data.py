import os
from collections import Counter

import datasets
import pandas as pd

# If you use proxy to download dataset from Hugging Face, plz:
os.environ["CURL_CA_BUNDLE"] = ""


def preprocess_file_paths(row, data_dir):
    if len(row["file_name"]) > 0:
        row["file_name"] = os.path.join(data_dir, row["file_name"])
    return row


def preprocess_data(ds, data_dir):
    ds = ds.rename_columns(
        {"Question": "question", "Final answer": "true_answer", "Level": "task"}
    )
    ds = ds.map(preprocess_file_paths, fn_kwargs={"data_dir": data_dir})
    df = pd.DataFrame(ds)
    print("Loaded dataset:")
    print(df["task"].value_counts())
    return df


# Load dataset, specify download mode
def load_gaia_data(
    data_dir: str = "data/",
    level: str = "level1",
    dataset: str = "validation",
):
    path = os.path.join(data_dir, f"gaia/{level}/{dataset}")
    # make sure the path exists
    os.makedirs(path, exist_ok=True)
    try:
        ds = datasets.load_from_disk(path)
        ds = preprocess_data(ds, data_dir)

    except FileNotFoundError:
        print(f"Dataset {path} not found, downloading from Hugging Face")
        # Make you have set the HF token in the environment variable
        ds = datasets.load_dataset(
            "gaia-benchmark/GAIA",
            f"2023_{level}",
            trust_remote_code=True,
            download_mode="force_redownload",
        )[dataset]

        # Save the dataset to a file
        ds.save_to_disk(path)
        ds = preprocess_data(ds, data_dir)
    return ds


def parse_tools(tools_list):
    """
    解析工具列表并生成工具集合统计

    Args:
        tools_list: 包含工具描述的字符串列表

    Returns:
        Dictionary: 包含所有工具及其出现频率
    """
    all_tools = []

    # 定义工具名称映射，将相似工具统一命名
    tool_mapping = {
        # 浏览器类
        "web browser": "web_browser",
        "a web browser": "web_browser",
        "a web browser.": "web_browser",
        # 搜索引擎类
        "search engine": "search_engine",
        "a search engine": "search_engine",
        "a search engine.": "search_engine",
        "google search": "search_engine",
        # 计算器类
        "calculator": "calculator",
        "a calculator": "calculator",
        "a calculator.": "calculator",
        "calculator (or ability to count)": "calculator",
        # 图像识别类
        "image recognition": "image_recognition",
        "image recognition tools": "image_recognition",
        "image recognition/ocr": "image_recognition",
        "color recognition": "image_recognition",
        # 文档查看类
        "pdf access": "document_viewer",
        "pdf viewer": "document_viewer",
        "word document access": "document_viewer",
        "text editor": "document_viewer",
        "powerpoint viewer": "document_viewer",
        "access to excel files": "document_viewer",
        "excel": "document_viewer",
        # 音频视频类
        "video parsing": "media_processor",
        "audio capability": "media_processor",
        "video processing software": "media_processor",
        "audio processing software": "media_processor",
        "video recognition tools": "media_processor",
        "a speech-to-text tool": "media_processor",
        "a speech-to-text audio processing tool": "media_processor",
        # 其他工具
        "a word reversal tool / script": "text_processor",
        "markdown": "text_processor",
        "no tools required": "none",
        "a file interface": "file_interface",
        "python": "programming",
        "access to academic journal websites": "academic_resources",
        "rubik's cube model": "special_tools",
        "wikipedia": "academic_resources",
    }

    # 工具分类的中文描述
    category_names = {
        "web_browser": "网页浏览器",
        "search_engine": "搜索引擎",
        "calculator": "计算器",
        "image_recognition": "图像识别",
        "document_viewer": "文档查看器",
        "media_processor": "媒体处理",
        "text_processor": "文本处理",
        "file_interface": "文件接口",
        "programming": "编程工具",
        "academic_resources": "学术资源",
        "special_tools": "特殊工具",
        "none": "无需工具",
    }

    for tools_str in tools_list:
        if tools_str == "None" or tools_str is None:
            continue

        # 按行分割
        tools = tools_str.split("\n")

        for tool in tools:
            # 移除数字编号和前导空格
            cleaned_tool = tool.strip()
            # 使用正则表达式或简单方法移除前面的数字和点
            if cleaned_tool and cleaned_tool[0].isdigit():
                # 找到第一个点后的内容
                if "." in cleaned_tool:
                    cleaned_tool = cleaned_tool.split(".", 1)[1].strip()

            if cleaned_tool and cleaned_tool.lower() != "none":
                tool_lower = cleaned_tool.lower()
                # 使用映射表规范化工具名称
                normalized_tool = tool_mapping.get(tool_lower, tool_lower)
                all_tools.append(normalized_tool)

    # 统计工具出现频率
    tool_counter = Counter(all_tools)

    # 按出现频率排序
    sorted_tools = dict(sorted(tool_counter.items(), key=lambda x: x[1], reverse=True))

    return sorted_tools, category_names


if __name__ == "__main__":
    ds = load_gaia_data()
    tools_data = [i["Tools"] for i in ds["Annotator Metadata"]]
    print("原始工具列表:")
    print(tools_data)

    # 解析工具集合
    tool_set, category_names = parse_tools(tools_data)
    print("\n工具集合统计:")
    for tool, count in tool_set.items():
        # 显示工具类别的中文名称
        tool_name = category_names.get(tool, tool)
        print(f"{tool_name} ({tool}): {count}次")
