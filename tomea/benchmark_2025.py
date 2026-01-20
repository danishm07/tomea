# benchmark_2025_fresh.py
"""
Fresh 2025 Papers Benchmark - Reality Check
Tests on papers from November-December 2025 with limited reference implementations
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from parallel import run_parallel_validation
import asyncio

# ============================================================================
# 2025 FRESH PAPERS (Nov-Dec 2025)
# ============================================================================

FRESH_2025_PAPERS = [
    # ========================================================================
    # CATEGORY 1: RECENT PEFT METHODS (Expected: 60-75% success)
    # ========================================================================
    {
        "name": "TS-PEFT (Token-Level PEFT)",
        "arxiv_id": "2511.16147",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "PEFT_2025",
        "difficulty": "hard",
        "notes": "Nov 2025 - Dynamic token-level parameter updates, very fresh paper"
    },
    {
        "name": "PEFT-Factory Framework",
        "arxiv_id": "2512.02764",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "PEFT_2025",
        "difficulty": "medium",
        "notes": "Dec 2025 - Framework paper, should have standard LoRA/BitFit implementations"
    },
    {
        "name": "PEFT-Bench Benchmark",
        "arxiv_id": "2511.21285",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "PEFT_2025",
        "difficulty": "medium",
        "notes": "Nov 2025 - Benchmark paper, evaluates existing PEFT methods"
    },
    
    # ========================================================================
    # CATEGORY 2: FINE-TUNING SAFETY & OPTIMIZATION (Expected: 40-60%)
    # ========================================================================
    {
        "name": "Dynamic Safety Shaping (STAR-DSS)",
        "arxiv_id": "2505.17196",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Safety_2025",
        "difficulty": "hard",
        "notes": "Oct 2025 - Safety-aware fine-tuning with dynamic shaping"
    },
    {
        "name": "Federated Fine-Tuning for MoE",
        "arxiv_id": "2508.19078",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Optimization_2025",
        "difficulty": "hard",
        "notes": "Oct 2025 - Distributed training, likely requires federated setup"
    },
    
    # ========================================================================
    # CATEGORY 3: LLM SURVEYS/REVIEWS (Expected: 30-50%)
    # ========================================================================
    {
        "name": "LLaMA Evolution Survey",
        "arxiv_id": "2510.12178",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Survey_2025",
        "difficulty": "medium",
        "notes": "Oct 2025 - Survey paper, should extract standard fine-tuning methods"
    },
    {
        "name": "Fine-Tuning Guide 2025",
        "arxiv_id": "2408.13296",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Survey_2025",
        "difficulty": "easy",
        "notes": "Sept 2025 - Comprehensive guide, tests extraction of standard methods"
    },
    
    # ========================================================================
    # CATEGORY 4: RECENT SENTIMENT/TEXT CLASSIFICATION (Expected: 50-70%)
    # ========================================================================
    {
        "name": "Zero-Label Sentiment (ESCS-GPT)",
        "arxiv_id": "2502.02893",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "TextClass_2025",
        "difficulty": "hard",
        "notes": "Feb 2025 - LLM-based labeling, requires GPT-based preprocessing"
    },
    {
        "name": "Sentiment with Emoji Data",
        "arxiv_id": "2502.13278",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "TextClass_2025",
        "difficulty": "medium",
        "notes": "Feb 2025 - Universal Sentence Encoder + BERT, tests embedding approaches"
    },
    {
        "name": "Three-Class LSTM Sentiment",
        "arxiv_id": "2412.17347",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "TextClass_2025",
        "difficulty": "easy",
        "notes": "Dec 2024 - Bi-GRU based, straightforward architecture"
    },
    
    # ========================================================================
    # CATEGORY 5: ADVANCED ARCHITECTURES (Expected: 20-40%)
    # ========================================================================
    {
        "name": "Nexus (Higher-Order Attention)",
        "arxiv_id": "2512.03377",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Architecture_2025",
        "difficulty": "hard",
        "notes": "Dec 2025 - Novel attention mechanism, likely very challenging"
    },
    {
        "name": "ScaleFormer (Long-Context)",
        "arxiv_id": "2511.10029",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Architecture_2025",
        "difficulty": "hard",
        "notes": "Nov 2025 - Long-context transformers, span representation"
    },
    {
        "name": "Non-Resolution Reasoning (NRR)",
        "arxiv_id": "2512.13478",
        "dataset_url": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
        "category": "Architecture_2025",
        "difficulty": "hard",
        "notes": "Dec 2025 - Multi-vector embeddings, theoretical architecture"
    },
]


# ============================================================================
# REALITY CHECK RUNNER
# ============================================================================

async def run_reality_check_benchmark(
    output_dir: str = "benchmark_2025_reality_check",
    max_attempts: int = 3
):
    """
    Run benchmark on fresh 2025 papers.
    This is the REAL test of the system's capabilities.
    """
    
    print("\n" + "="*80)
    print("🔬 2025 REALITY CHECK BENCHMARK")
    print("="*80)
    print(f"\nTesting on {len(FRESH_2025_PAPERS)} papers from Nov-Dec 2025")
    print("These papers have MINIMAL reference implementations")
    print("This is the true test of code generation quality\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Categorize papers
    categories = {}
    for paper in FRESH_2025_PAPERS:
        cat = paper["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(paper)
    
    print("📊 Paper Distribution:")
    for cat, papers in categories.items():
        print(f"   {cat:20s}: {len(papers)} papers")
    print()
    
    # Run benchmark using parallel validation
    start_time = time.time()
    
    dataset_url = FRESH_2025_PAPERS[0]["dataset_url"]  # All use IMDB for now
    
    results = await run_parallel_validation(
        papers=[{"arxiv": p["arxiv_id"], "name": p["name"]} for p in FRESH_2025_PAPERS],
        dataset_url=dataset_url,
        max_attempts=max_attempts
    )
    
    total_time = time.time() - start_time
    
    # Analyze results
    analysis = analyze_results(results, FRESH_2025_PAPERS)
    
    # Save detailed results
    save_detailed_results(analysis, output_dir)
    
    # Print summary
    print_reality_check_summary(analysis, total_time)
    
    return analysis


def analyze_results(results: List, papers: List[Dict]) -> Dict:
    """Analyze benchmark results by category and difficulty"""
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_papers": len(papers),
        "successes": sum(1 for r in results if r.success),
        "failures": sum(1 for r in results if not r.success),
        "overall_success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
        "by_category": {},
        "by_difficulty": {},
        "failed_papers": [],
        "successful_papers": [],
        "top_errors": {}
    }
    
    # Map results to paper metadata
    paper_map = {p["name"]: p for p in papers}
    
    for result in results:
        paper_info = paper_map.get(result.paper_name, {})
        category = paper_info.get("category", "Unknown")
        difficulty = paper_info.get("difficulty", "unknown")
        
        # By category
        if category not in analysis["by_category"]:
            analysis["by_category"][category] = {
                "total": 0,
                "success": 0,
                "failure": 0,
                "papers": []
            }
        
        analysis["by_category"][category]["total"] += 1
        if result.success:
            analysis["by_category"][category]["success"] += 1
        else:
            analysis["by_category"][category]["failure"] += 1
        
        analysis["by_category"][category]["papers"].append({
            "name": result.paper_name,
            "success": result.success,
            "attempts": result.attempts
        })
        
        # By difficulty
        if difficulty not in analysis["by_difficulty"]:
            analysis["by_difficulty"][difficulty] = {
                "total": 0,
                "success": 0,
                "failure": 0
            }
        
        analysis["by_difficulty"][difficulty]["total"] += 1
        if result.success:
            analysis["by_difficulty"][difficulty]["success"] += 1
        else:
            analysis["by_difficulty"][difficulty]["failure"] += 1
        
        # Track failures
        if not result.success:
            analysis["failed_papers"].append({
                "name": result.paper_name,
                "arxiv_id": result.arxiv_id,
                "category": category,
                "difficulty": difficulty,
                "attempts": result.attempts,
                "errors": result.error_log[-3:] if result.error_log else []
            })
        else:
            analysis["successful_papers"].append({
                "name": result.paper_name,
                "arxiv_id": result.arxiv_id,
                "attempts": result.attempts,
                "final_loss": result.final_loss,
                "final_eval_loss": result.final_eval_loss
            })
    
    # Calculate success rates
    for cat_data in analysis["by_category"].values():
        cat_data["success_rate"] = cat_data["success"] / cat_data["total"] if cat_data["total"] > 0 else 0
    
    for diff_data in analysis["by_difficulty"].values():
        diff_data["success_rate"] = diff_data["success"] / diff_data["total"] if diff_data["total"] > 0 else 0
    
    return analysis


def save_detailed_results(analysis: Dict, output_dir: str):
    """Save detailed analysis to JSON"""
    filename = Path(output_dir) / f"reality_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n💾 Detailed results saved to: {filename}")


def print_reality_check_summary(analysis: Dict, total_time: float):
    """Print comprehensive summary"""
    
    print("\n" + "="*80)
    print("📊 REALITY CHECK RESULTS")
    print("="*80 + "\n")
    
    # Overall
    print(f"{'Overall Performance':30s}")
    print(f"{'─'*30}")
    print(f"  Total Papers:        {analysis['total_papers']}")
    print(f"  ✅ Successes:        {analysis['successes']} ({analysis['overall_success_rate']*100:.1f}%)")
    print(f"  ❌ Failures:         {analysis['failures']} ({(1-analysis['overall_success_rate'])*100:.1f}%)")
    print(f"  ⏱️  Total Time:       {total_time/60:.1f} minutes")
    print()
    
    # Interpretation
    success_rate = analysis['overall_success_rate']
    if success_rate >= 0.70:
        verdict = "🎉 EXCELLENT - System handles fresh papers well!"
        color = "green"
    elif success_rate >= 0.50:
        verdict = "✅ GOOD - Decent performance, some improvements needed"
        color = "yellow"
    elif success_rate >= 0.30:
        verdict = "⚠️  FAIR - Significant work needed on error handling"
        color = "yellow"
    else:
        verdict = "❌ NEEDS WORK - Core improvements required"
        color = "red"
    
    print(f"{'Verdict':30s}")
    print(f"{'─'*30}")
    print(f"  {verdict}")
    print()
    
    # By Category
    print(f"{'Performance by Category':30s}")
    print(f"{'─'*30}")
    for category, data in sorted(analysis["by_category"].items()):
        rate = data["success_rate"] * 100
        status = "✅" if rate >= 60 else "⚠️" if rate >= 40 else "❌"
        print(f"  {status} {category:25s}: {data['success']:2d}/{data['total']:2d} ({rate:5.1f}%)")
    print()
    
    # By Difficulty
    print(f"{'Performance by Difficulty':30s}")
    print(f"{'─'*30}")
    for difficulty in ["easy", "medium", "hard"]:
        if difficulty in analysis["by_difficulty"]:
            data = analysis["by_difficulty"][difficulty]
            rate = data["success_rate"] * 100
            print(f"  {difficulty.capitalize():10s}: {data['success']:2d}/{data['total']:2d} ({rate:5.1f}%)")
    print()
    
    # Top Failures
    if analysis["failed_papers"]:
        print(f"{'Failed Papers (Investigate These)':30s}")
        print(f"{'─'*30}")
        for paper in analysis["failed_papers"][:5]:  # Top 5
            print(f"  ❌ {paper['name']}")
            print(f"     Category: {paper['category']}, Difficulty: {paper['difficulty']}")
            print(f"     ArXiv: {paper['arxiv_id']}, Attempts: {paper['attempts']}")
            if paper['errors']:
                print(f"     Last error: {paper['errors'][-1][:80]}...")
            print()
    
    # Next Steps
    print(f"{'Recommended Next Steps':30s}")
    print(f"{'─'*30}")
    
    if success_rate >= 0.70:
        print("  1. ✅ System is production-ready for this paper type")
        print("  2. 📈 Expand to more diverse tasks (summarization, QA)")
        print("  3. 👥 Get 5-10 users to test")
    elif success_rate >= 0.50:
        print("  1. 🔍 Analyze top 3 failure modes")
        print("  2. 🛠️  Add deterministic fixes for those errors")
        print("  3. 🔄 Re-run benchmark")
    else:
        print("  1. ❗ Focus on error classification improvements")
        print("  2. 🐛 Fix critical bugs before adding features")
        print("  3. 📝 Document common failure patterns")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# COMPARISON MODE
# ============================================================================

async def compare_old_vs_new_benchmark(
    old_results_file: str = "benchmark_results/summary_latest.json",
    max_attempts: int = 3
):
    """
    Compare performance on old benchmark vs fresh 2025 papers.
    This shows if the system is overfit to papers with lots of reference code.
    """
    
    print("\n" + "="*80)
    print("📊 OLD vs NEW BENCHMARK COMPARISON")
    print("="*80 + "\n")
    
    # Run new benchmark
    new_analysis = await run_reality_check_benchmark(
        output_dir="benchmark_2025_reality_check",
        max_attempts=max_attempts
    )
    
    # Load old results if available
    try:
        with open(old_results_file, 'r') as f:
            old_results = json.load(f)
        old_success_rate = old_results["summary"]["success_rate"]
        
        print(f"\n{'Comparison':30s}")
        print(f"{'─'*30}")
        print(f"  Old Benchmark (2022-2024):  {old_success_rate*100:.1f}%")
        print(f"  New Benchmark (2025):        {new_analysis['overall_success_rate']*100:.1f}%")
        
        diff = (new_analysis['overall_success_rate'] - old_success_rate) * 100
        if abs(diff) < 5:
            print(f"  Difference:                  {diff:+.1f}% (Consistent! ✅)")
        elif diff < 0:
            print(f"  Difference:                  {diff:+.1f}% (Regression ⚠️)")
        else:
            print(f"  Difference:                  {diff:+.1f}% (Improvement! 🎉)")
        
    except FileNotFoundError:
        print("⚠️  No old benchmark results found for comparison")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="2025 Reality Check Benchmark")
    parser.add_argument("--compare", action="store_true", help="Compare with old benchmark")
    parser.add_argument("--subset", type=int, help="Run on first N papers only")
    parser.add_argument("--category", type=str, help="Run only papers from this category")
    parser.add_argument("--attempts", type=int, default=3, help="Max attempts per paper")
    
    args = parser.parse_args()
    
    # Filter papers if needed
    papers_to_run = FRESH_2025_PAPERS
    if args.category:
        papers_to_run = [p for p in papers_to_run if p["category"] == args.category]
        print(f"🔍 Filtering to category: {args.category} ({len(papers_to_run)} papers)")
    if args.subset:
        papers_to_run = papers_to_run[:args.subset]
        print(f"🔍 Running subset: first {len(papers_to_run)} papers")
    
    # Update global list
    FRESH_2025_PAPERS = papers_to_run
    
    if args.compare:
        asyncio.run(compare_old_vs_new_benchmark(max_attempts=args.attempts))
    else:
        asyncio.run(run_reality_check_benchmark(max_attempts=args.attempts))