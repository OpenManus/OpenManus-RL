#!/usr/bin/env python3
"""
Together AI 并行调用性能测试脚本
支持并行调用，测试模型性能和响应时间
"""

import asyncio
import time
import statistics
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from openai import OpenAI


class TogetherAITester:
    """Together AI 性能测试器"""
    
    def __init__(
        self,
        model_name: str = "kunlunz2/Qwen/Qwen3-8B-9f9838eb",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        default_base_url = os.environ.get("TOGETHER_API_BASE_URL", "https://api.together.xyz/v1")
        resolved_base_url = base_url or default_base_url
        self.client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url=resolved_base_url,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.test_messages = [
            {"role": "user", "content": "What are some fun things to do in New York?"},
            {"role": "user", "content": "Explain quantum computing in simple terms."},
            {"role": "user", "content": "Write a short poem about artificial intelligence."},
            {"role": "user", "content": "What is the capital of France and why is it important?"},
            {"role": "user", "content": "Calculate the factorial of 10."},
            {"role": "user", "content": "Describe the process of photosynthesis."},
            {"role": "user", "content": "What are the benefits of renewable energy?"},
            {"role": "user", "content": "Explain the theory of relativity briefly."},
            {"role": "user", "content": "How does machine learning work?"},
            {"role": "user", "content": "What are the main causes of climate change?"}
        ]
    
    def single_request(self, request_id: int) -> Dict[str, Any]:
        """执行单次请求"""
        start_time = time.time()
        success = False
        error_msg = None
        response_length = 0
        
        try:
            # 循环使用不同的测试消息
            message = self.test_messages[request_id % len(self.test_messages)]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[message],
                max_tokens=150,
                temperature=self.temperature,
            )
            
            if response and response.choices:
                content = response.choices[0].message.content
                response_length = len(content)
                success = True
            else:
                error_msg = "Empty response"
                
        except Exception as e:
            error_msg = str(e)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'request_id': request_id,
            'success': success,
            'duration': duration,
            'response_length': response_length,
            'error': error_msg,
            'timestamp': start_time
        }
    
    def run_parallel_test(self, total_requests: int = 100, max_workers: int = 10) -> Dict[str, Any]:
        """运行并行测试"""
        print(f"🚀 开始并行测试:")
        print(f"   模型: {self.model_name}")
        print(f"   Temperature: {self.temperature}")
        print(f"   总请求数: {total_requests}")
        print(f"   并行数: {max_workers}")
        print(f"   {'='*50}")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.single_request, i): i 
                for i in range(total_requests)
            }
            
            # 收集结果并显示进度
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                # 每10个请求显示一次进度
                if completed % 10 == 0 or completed == total_requests:
                    success_count = sum(1 for r in results if r['success'])
                    print(f"   进度: {completed}/{total_requests} "
                          f"(成功: {success_count}, 失败: {completed - success_count})")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        return self.analyze_results(results, total_duration)
    
    def analyze_results(self, results: List[Dict[str, Any]], total_duration: float) -> Dict[str, Any]:
        """分析测试结果"""
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if not successful_results:
            print("❌ 所有请求都失败了!")
            return {'success': False, 'error': 'All requests failed'}
        
        # 计算统计数据
        durations = [r['duration'] for r in successful_results]
        response_lengths = [r['response_length'] for r in successful_results]
        
        stats = {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'total_duration': total_duration,
            'requests_per_second': len(results) / total_duration,
            'successful_rps': len(successful_results) / total_duration,
            'avg_response_time': statistics.mean(durations),
            'median_response_time': statistics.median(durations),
            'min_response_time': min(durations),
            'max_response_time': max(durations),
            'std_response_time': statistics.stdev(durations) if len(durations) > 1 else 0,
            'avg_response_length': statistics.mean(response_lengths),
            'total_tokens': sum(response_lengths)
        }
        
        return stats
    
    def print_results(self, stats: Dict[str, Any]):
        """打印测试结果"""
        if not stats.get('success', True):
            print(f"❌ 测试失败: {stats.get('error', 'Unknown error')}")
            return
        
        print(f"\n📊 测试结果统计:")
        print(f"{'='*50}")
        print(f"总体统计:")
        print(f"  • 总请求数: {stats['total_requests']}")
        print(f"  • 成功请求: {stats['successful_requests']}")
        print(f"  • 失败请求: {stats['failed_requests']}")
        print(f"  • 成功率: {stats['success_rate']:.2f}%")
        print(f"  • 总耗时: {stats['total_duration']:.2f} 秒")
        
        print(f"\n性能指标:")
        print(f"  • 总请求速率: {stats['requests_per_second']:.2f} RPS")
        print(f"  • 成功请求速率: {stats['successful_rps']:.2f} RPS")
        
        print(f"\n响应时间统计:")
        print(f"  • 平均响应时间: {stats['avg_response_time']:.3f} 秒")
        print(f"  • 中位响应时间: {stats['median_response_time']:.3f} 秒")
        print(f"  • 最快响应时间: {stats['min_response_time']:.3f} 秒")
        print(f"  • 最慢响应时间: {stats['max_response_time']:.3f} 秒")
        print(f"  • 响应时间标准差: {stats['std_response_time']:.3f} 秒")
        
        print(f"\n内容统计:")
        print(f"  • 平均响应长度: {stats['avg_response_length']:.0f} 字符")
        print(f"  • 总响应字符数: {stats['total_tokens']}")
        
        # 性能评估
        print(f"\n🎯 性能评估:")
        if stats['success_rate'] >= 95:
            print("  ✅ 成功率: 优秀")
        elif stats['success_rate'] >= 90:
            print("  ⚠️  成功率: 良好")
        else:
            print("  ❌ 成功率: 需要改进")
        
        if stats['avg_response_time'] <= 2.0:
            print("  ✅ 响应速度: 优秀")
        elif stats['avg_response_time'] <= 5.0:
            print("  ⚠️  响应速度: 良好")
        else:
            print("  ❌ 响应速度: 需要改进")
        
        if stats['successful_rps'] >= 5:
            print("  ✅ 吞吐量: 优秀")
        elif stats['successful_rps'] >= 2:
            print("  ⚠️  吞吐量: 良好")
        else:
            print("  ❌ 吞吐量: 需要改进")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Together AI 并行调用性能测试")
    parser.add_argument(
        "--model",
        default="kunlunz2/Qwen/Qwen3-8B-9f9838eb",
        help=(
            "要测试的模型名称 (示例: kunlunz2/Qwen/Qwen3-8B-9f9838eb)"
        ),
    )
    parser.add_argument("--requests", type=int, default=100,
                       help="总请求数量 (默认: 100)")
    parser.add_argument("--parallel", type=int, default=10,
                       help="并行数量 (默认: 10)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="采样温度 (默认: 0.0)")
    parser.add_argument("--base-url", dest="base_url", default=None,
                       help="Together API Base URL (默认: 环境变量 TOGETHER_API_BASE_URL 或 https://api.together.xyz/v1)")
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式 (10次请求)")
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick:
        args.requests = 10
        args.parallel = 3
    
    tester = TogetherAITester(
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    
    try:
        stats = tester.run_parallel_test(
            total_requests=args.requests,
            max_workers=args.parallel
        )
        tester.print_results(stats)
        
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")


if __name__ == "__main__":
    main()
