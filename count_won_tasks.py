#!/usr/bin/env python3
"""
统计指定文件夹及其子文件夹中所有 task_summary.json 文件中 won 为 true 的任务数量
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def find_task_summary_files(root_folder: str) -> List[str]:
    """
    递归查找指定文件夹及其子文件夹中的所有 task_summary.json 文件
    
    Args:
        root_folder: 根文件夹路径
        
    Returns:
        task_summary.json 文件路径列表
    """
    task_files = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == 'task_summary.json':
                file_path = os.path.join(root, file)
                task_files.append(file_path)
    
    return task_files


def parse_task_summary(file_path: str) -> Dict[str, Any]:
    """
    解析 task_summary.json 文件
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        解析后的 JSON 数据，如果解析失败返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"警告: 无法解析文件 {file_path}: {e}")
        return None


def count_won_tasks(root_folder: str) -> Dict[str, Any]:
    """
    统计 won 为 true 的任务数量
    
    Args:
        root_folder: 根文件夹路径
        
    Returns:
        包含统计结果的字典
    """
    task_files = find_task_summary_files(root_folder)
    
    total_files = len(task_files)
    won_count = 0
    lost_count = 0
    invalid_files = 0
    
    won_files = []
    lost_files = []
    
    print(f"找到 {total_files} 个 task_summary.json 文件")
    print("正在统计...")
    
    for file_path in task_files:
        data = parse_task_summary(file_path)
        
        if data is None:
            invalid_files += 1
            continue
            
        # 检查 metadata 中的 won 字段
        metadata = data.get('metadata', {})
        won = metadata.get('won', False)
        
        if won:
            won_count += 1
            won_files.append(file_path)
        else:
            lost_count += 1
            lost_files.append(file_path)
    
    return {
        'total_files': total_files,
        'won_count': won_count,
        'lost_count': lost_count,
        'invalid_files': invalid_files,
        'won_files': won_files,
        'lost_files': lost_files,
        'success_rate': (won_count / (won_count + lost_count) * 100) if (won_count + lost_count) > 0 else 0
    }


def print_statistics(stats: Dict[str, Any], verbose: bool = False):
    """
    打印统计结果
    
    Args:
        stats: 统计结果字典
        verbose: 是否显示详细信息
    """
    print("\n" + "="*50)
    print("统计结果")
    print("="*50)
    print(f"总文件数量: {stats['total_files']}")
    print(f"成功任务 (won=true): {stats['won_count']}")
    print(f"失败任务 (won=false): {stats['lost_count']}")
    print(f"无效文件数量: {stats['invalid_files']}")
    print(f"成功率: {stats['success_rate']:.2f}%")
    
    if verbose:
        print("\n" + "-"*30)
        print("成功任务文件列表:")
        for file_path in stats['won_files']:
            print(f"  {file_path}")
        
        print("\n" + "-"*30)
        print("失败任务文件列表:")
        for file_path in stats['lost_files']:
            print(f"  {file_path}")


def main():
    parser = argparse.ArgumentParser(description='统计 task_summary.json 文件中 won=true 的任务数量')
    parser.add_argument('--folder', default='experiments/table2/webshop_qwen3-8b_self_refine_maxtry5_env50_start1_20250923_172036', help='要统计的文件夹路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"错误: 文件夹 '{args.folder}' 不存在")
        return
    
    if not os.path.isdir(args.folder):
        print(f"错误: '{args.folder}' 不是一个文件夹")
        return
    
    print(f"正在分析文件夹: {args.folder}")
    
    stats = count_won_tasks(args.folder)
    print_statistics(stats, args.verbose)


if __name__ == "__main__":
    main()
