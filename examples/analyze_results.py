#!/usr/bin/env python3
"""
Analysis tool for evaluation results
Provides insights and visualizations from trajectory evaluations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
from collections import defaultdict


class ResultsAnalyzer:
    """Analyze evaluation results to identify patterns and insights"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = self._load_all_results()
    
    def _load_all_results(self) -> List[Dict]:
        """Load all evaluation result files"""
        results = []
        for eval_file in self.results_dir.glob("eval_*.json"):
            try:
                with open(eval_file) as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"Error loading {eval_file}: {e}")
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        
        if not self.data:
            return {"error": "No evaluation data found"}
        
        report = {
            "overview": self._get_overview(),
            "module_analysis": self._analyze_modules(),
            "failure_analysis": self._analyze_failures(),
            "success_patterns": self._identify_success_patterns(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _get_overview(self) -> Dict:
        """Generate overview statistics"""
        
        total = len(self.data)
        successful = sum(1 for d in self.data if d['success'])
        scores = [d['overall_score'] for d in self.data]
        
        return {
            "total_trajectories": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_score": statistics.mean(scores) if scores else 0,
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0
        }
    
    def _analyze_modules(self) -> Dict:
        """Analyze performance by module"""
        
        module_stats = defaultdict(lambda: {"scores": [], "issues": []})
        
        for trajectory in self.data:
            for step_eval in trajectory.get('step_evaluations', []):
                for module_name, module_data in step_eval.get('module_scores', {}).items():
                    score = module_data['score']
                    module_stats[module_name]['scores'].append(score)
                    
                    if score <= 2:  # Poor performance
                        module_stats[module_name]['issues'].append({
                            'task_id': trajectory['task_id'],
                            'step': step_eval['step_num'],
                            'score': score,
                            'reasoning': module_data['reasoning'][:200]
                        })
        
        # Calculate statistics for each module
        analysis = {}
        for module_name, data in module_stats.items():
            scores = data['scores']
            if scores:
                analysis[module_name] = {
                    'average_score': statistics.mean(scores),
                    'median_score': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'total_evaluations': len(scores),
                    'poor_performance_count': len(data['issues']),
                    'poor_performance_rate': len(data['issues']) / len(scores),
                    'worst_cases': sorted(data['issues'], key=lambda x: x['score'])[:3]
                }
        
        return analysis
    
    def _analyze_failures(self) -> Dict:
        """Analyze failed trajectories to identify patterns"""
        
        failed_trajectories = [d for d in self.data if not d['success']]
        
        if not failed_trajectories:
            return {"message": "No failed trajectories found"}
        
        # Collect failure patterns
        weakness_counts = defaultdict(int)
        critical_issue_counts = defaultdict(int)
        low_scoring_modules = defaultdict(list)
        
        for trajectory in failed_trajectories:
            # Count weaknesses
            for weakness in trajectory.get('weaknesses', []):
                weakness_counts[weakness] += 1
            
            # Count critical issues
            for step_eval in trajectory.get('step_evaluations', []):
                for issue in step_eval.get('critical_issues', []):
                    issue_type = issue.split(':')[0] if ':' in issue else issue
                    critical_issue_counts[issue_type] += 1
                
                # Track low-scoring modules
                for module_name, module_data in step_eval.get('module_scores', {}).items():
                    if module_data['score'] <= 1:
                        low_scoring_modules[module_name].append({
                            'task_id': trajectory['task_id'],
                            'score': module_data['score']
                        })
        
        return {
            'total_failures': len(failed_trajectories),
            'average_failure_score': statistics.mean([t['overall_score'] for t in failed_trajectories]),
            'common_weaknesses': dict(sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'critical_issues': dict(sorted(critical_issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'problematic_modules': {
                module: {
                    'failure_count': len(cases),
                    'average_score': statistics.mean([c['score'] for c in cases])
                }
                for module, cases in low_scoring_modules.items()
            }
        }
    
    def _identify_success_patterns(self) -> Dict:
        """Identify patterns in successful trajectories"""
        
        successful_trajectories = [d for d in self.data if d['success']]
        
        if not successful_trajectories:
            return {"message": "No successful trajectories found"}
        
        # Analyze successful patterns
        strength_counts = defaultdict(int)
        high_scoring_modules = defaultdict(list)
        
        for trajectory in successful_trajectories:
            # Count strengths
            for strength in trajectory.get('strengths', []):
                strength_counts[strength] += 1
            
            # Track high-scoring modules
            for step_eval in trajectory.get('step_evaluations', []):
                for module_name, module_data in step_eval.get('module_scores', {}).items():
                    if module_data['score'] >= 4:
                        high_scoring_modules[module_name].append(module_data['score'])
        
        return {
            'total_successes': len(successful_trajectories),
            'average_success_score': statistics.mean([t['overall_score'] for t in successful_trajectories]),
            'common_strengths': dict(sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'strong_modules': {
                module: {
                    'high_score_count': len(scores),
                    'average_score': statistics.mean(scores)
                }
                for module, scores in high_scoring_modules.items()
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        module_analysis = self._analyze_modules()
        failure_analysis = self._analyze_failures()
        
        # Identify weakest modules
        weakest_modules = sorted(
            module_analysis.items(),
            key=lambda x: x[1]['average_score']
        )[:2]
        
        for module_name, stats in weakest_modules:
            if stats['average_score'] < 3:
                recommendations.append(
                    f"CRITICAL: {module_name.replace('_', ' ').title()} module needs immediate improvement "
                    f"(avg score: {stats['average_score']:.1f}, {stats['poor_performance_rate']:.0%} poor performance rate)"
                )
        
        # Check for systematic failures
        if failure_analysis.get('total_failures', 0) > 0:
            problematic = failure_analysis.get('problematic_modules', {})
            if problematic:
                worst_module = max(problematic.items(), key=lambda x: x[1]['failure_count'])
                recommendations.append(
                    f"Focus training on {worst_module[0]} module - involved in {worst_module[1]['failure_count']} failures"
                )
        
        # Success pattern recommendations
        success_patterns = self._identify_success_patterns()
        if success_patterns.get('strong_modules'):
            strong_modules = list(success_patterns['strong_modules'].keys())
            if strong_modules:
                recommendations.append(
                    f"Leverage strengths in {', '.join(strong_modules[:2])} modules as examples for improvement"
                )
        
        # General recommendations based on scores
        overview = self._get_overview()
        if overview['average_score'] < 3:
            recommendations.append(
                "Overall performance is below adequate - consider comprehensive retraining"
            )
        elif overview['score_std'] > 1.5:
            recommendations.append(
                "High variance in performance - focus on consistency improvements"
            )
        
        if overview['success_rate'] < 0.5:
            recommendations.append(
                f"Success rate is only {overview['success_rate']:.0%} - prioritize task completion strategies"
            )
        
        return recommendations
    
    def print_report(self, report: Dict):
        """Print formatted analysis report"""
        
        print("\n" + "="*70)
        print("TRAJECTORY EVALUATION ANALYSIS REPORT")
        print("="*70)
        
        # Overview
        overview = report['overview']
        print("\n## OVERVIEW")
        print(f"Total Trajectories: {overview['total_trajectories']}")
        print(f"Success Rate: {overview['success_rate']:.1%} ({overview['successful']}/{overview['total_trajectories']})")
        print(f"Average Score: {overview['average_score']:.2f} ± {overview['score_std']:.2f}")
        print(f"Score Range: {overview['min_score']:.2f} - {overview['max_score']:.2f}")
        
        # Module Analysis
        print("\n## MODULE PERFORMANCE")
        module_analysis = report['module_analysis']
        for module_name, stats in sorted(module_analysis.items(), key=lambda x: x[1]['average_score']):
            print(f"\n{module_name.replace('_', ' ').title()}:")
            print(f"  Average Score: {stats['average_score']:.2f}")
            print(f"  Poor Performance Rate: {stats['poor_performance_rate']:.1%}")
            print(f"  Total Evaluations: {stats['total_evaluations']}")
        
        # Failure Analysis
        if report['failure_analysis'].get('total_failures'):
            print("\n## FAILURE ANALYSIS")
            failure_data = report['failure_analysis']
            print(f"Failed Trajectories: {failure_data['total_failures']}")
            print(f"Average Failure Score: {failure_data['average_failure_score']:.2f}")
            
            if failure_data.get('common_weaknesses'):
                print("\nMost Common Weaknesses:")
                for weakness, count in list(failure_data['common_weaknesses'].items())[:3]:
                    print(f"  - {weakness}: {count} occurrences")
        
        # Success Patterns
        if report['success_patterns'].get('total_successes'):
            print("\n## SUCCESS PATTERNS")
            success_data = report['success_patterns']
            print(f"Successful Trajectories: {success_data['total_successes']}")
            print(f"Average Success Score: {success_data['average_success_score']:.2f}")
            
            if success_data.get('common_strengths'):
                print("\nMost Common Strengths:")
                for strength, count in list(success_data['common_strengths'].items())[:3]:
                    print(f"  - {strength}: {count} occurrences")
        
        # Recommendations
        print("\n## RECOMMENDATIONS")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*70)
    
    def save_report(self, report: Dict, output_file: str = "analysis_report.json"):
        """Save analysis report to file"""
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_path}")
    
    def export_training_data(self) -> List[Dict]:
        """Export data in format suitable for training"""
        
        training_data = []
        
        for trajectory in self.data:
            for step_eval in trajectory.get('step_evaluations', []):
                for module_name, module_data in step_eval.get('module_scores', {}).items():
                    training_data.append({
                        'task_id': trajectory['task_id'],
                        'task_description': trajectory['task_description'],
                        'step_num': step_eval['step_num'],
                        'module': module_name,
                        'score': module_data['score'],
                        'success': trajectory['success'],
                        'suggestions': module_data.get('suggestions', [])
                    })
        
        return training_data


def main():
    """Main analysis function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trajectory evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../evaluation_results",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--export-training",
        action="store_true",
        help="Export data for training"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save analysis report to file"
    )
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not Path(args.results_dir).exists():
        print(f"Error: Results directory '{args.results_dir}' not found")
        print("Please run evaluation first: python run_evaluation.py")
        return
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if not analyzer.data:
        print("No evaluation data found in the results directory")
        return
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print report
    analyzer.print_report(report)
    
    # Save report if requested
    if args.save_report:
        analyzer.save_report(report)
    
    # Export training data if requested
    if args.export_training:
        training_data = analyzer.export_training_data()
        output_file = Path(args.results_dir) / "training_data.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\nTraining data exported to: {output_file}")
        print(f"Total training samples: {len(training_data)}")


if __name__ == "__main__":
    main()