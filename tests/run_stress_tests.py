#!/usr/bin/env python3
"""
TOMEA V2 STRESS TEST SUITE
============================
Execution order: Test 4 (regression) → Test 1 (Dr. GRPO) → Test 2 (DAPO) → Test 3 (Kimi K2.5)

Run:
    python run_stress_tests.py          # all four
    python run_stress_tests.py --test 4 # single test by number
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime
import dotenv

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Add project root so `tomea.*` imports resolve
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tomea.core.orchestrator import TomeaOrchestrator

# ---------------------------------------------------------------------------
# Test definitions — single source of truth for what gets run
# ---------------------------------------------------------------------------
TESTS = {
    4: {
        "name": "DPO Baseline Regression",
        "arxiv_id": "2305.18290",
        "paper_text": "",              # empty → orchestrator fetches via get_paper_data() before routing
        "dataset": None,               # None → DataEngine defaults to Anthropic/hh-rlhf
        "description": (
            "Sanity gate. Runs FIRST. If this breaks, the V2 refactor "
            "regressed something fundamental and Tests 1-3 are meaningless. "
            "Same DPO smoke test that passed on V1."
        ),
    },
    1: {
        "name": "Dr. GRPO — Subtle Math",
        "arxiv_id": "2503.20783",
        "paper_text": "",
        "dataset": None,               # hh-rlhf default is fine; we're testing loss math, not data
        "description": (
            "Tests whether extraction + synthesis can detect a SUBTRACTIVE "
            "contribution. The novelty is two lines REMOVED from standard GRPO "
            "(length normalization and std normalization). If synthesizer "
            "pattern-matches on 'GRPO paper' and generates vanilla GRPO, "
            "the code still runs and produces a real loss — nobody knows it's wrong "
            "unless extraction read the paper carefully enough to name the specific removals."
        ),
    },
    2: {
        "name": "DAPO — Training Loop",
        "arxiv_id": "2503.14476",
        "paper_text": "",
        "dataset": None,
        "description": (
            "Tests whether the pipeline can handle contributions that break out "
            "of compute_loss entirely. Asymmetric clipping and token-level loss "
            "fit inside the template. Dynamic sampling does NOT — it's pre-optimizer "
            "batch filtering with no injection point in RLHarness. Right answer: "
            "extraction flags dynamic sampling as out-of-scope, implements what it can."
        ),
    },
    3: {
        "name": "Kimi K2.5 PARL — Missing Information",
        "arxiv_id": None,              # No arXiv paper exists for this
        "paper_text": "",              # Populated at runtime from the blog
        "dataset": None,
        "description": (
            "Tests whether the system is HONEST about its limits. K2.5's novel "
            "contribution (PARL) has no published loss function, no pseudocode, "
            "no algorithm specification — only a blog describing behavior. "
            "There is literally nothing to implement. Right answer: low extraction "
            "confidence, smoke_test_feasible = 'no'. High confidence = hallucination."
        ),
        "blog_url": "https://www.kimi.com/blog/kimi-k2-5.html",
    },
}

# Execution order
EXECUTION_ORDER = [4, 1, 2, 3]


# ---------------------------------------------------------------------------
# K2.5 blog text loader (Test 3 only — no PDF, no arXiv)
# ---------------------------------------------------------------------------
def load_kimi_blog_text(url: str) -> str:
    """
    Fetches the K2.5 blog page and strips it to relevant text.
    We do this at runtime so we don't hardcode stale content.
    """
    import urllib.request
    import re

    print(f"   📥 Fetching K2.5 blog from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8")

        # Strip tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 200:
            raise ValueError("Blog page returned too little text — possible block or layout change.")

        print(f"   ✅ Loaded {len(text)} chars from blog.")
        return text

    except Exception as e:
        print(f"   ❌ Blog fetch failed: {e}")
        print("   ℹ️  Falling back to cached PARL description.")
        # Minimal fallback so the test can still run and measure extraction behavior
        return (
            "Kimi K2.5 is trained with Parallel-Agent Reinforcement Learning (PARL). "
            "PARL uses a trainable orchestrator agent to decompose tasks into parallelizable "
            "subtasks, each executed by dynamically instantiated frozen subagents. "
            "K2.5 Agent Swarm can self-direct up to 100 sub-agents executing parallel "
            "workflows across up to 1500 coordinated steps. Training a reliable parallel "
            "orchestrator is challenging due to delayed, sparse, and non-stationary feedback "
            "from independently running subagents. A common failure mode is serial collapse "
            "where the orchestrator defaults to single-agent execution despite having parallel capacity. "
            "The reward increases smoothly as training progresses while the level of parallelism "
            "during training also gradually increases. No loss function derivation, no pseudocode, "
            "and no training algorithm specification are publicly available for PARL."
        )


# ---------------------------------------------------------------------------
# Result analysis — reads extraction_analysis.json after each run
# ---------------------------------------------------------------------------
def analyze_result(test_num: int, test_def: dict, run_dir: str, mission_success: bool) -> dict:
    """
    Post-run analysis. Reads the extraction metadata that the orchestrator
    saved and checks it against what we expect for this test.
    """
    result = {
        "test": test_num,
        "name": test_def["name"],
        "mission_success": mission_success,
        "extraction": None,
        "verdict": "UNKNOWN",
        "notes": [],
    }

    # Load extraction metadata
    extraction_path = os.path.join(run_dir, "extraction_analysis.json")
    if os.path.exists(extraction_path):
        with open(extraction_path) as f:
            result["extraction"] = json.load(f)
    else:
        result["notes"].append("⚠️  extraction_analysis.json not found — orchestrator may have crashed before synthesis.")
        result["verdict"] = "INFRASTRUCTURE_FAIL"
        return result

    ext = result["extraction"]
    confidence = ext.get("confidence", 0)
    feasible = ext.get("smoke_test_feasible", "unknown")
    contribution = ext.get("novel_contribution", "").lower()
    key_math = ext.get("key_math", "").lower()

    # ---------------------------------------------------------------
    # Test-specific verdict logic
    # ---------------------------------------------------------------
    if test_num == 4:
        # Regression: just needs to run clean
        result["verdict"] = "PASS" if mission_success else "FAIL"

    elif test_num == 1:
        # Dr. GRPO: extraction must name BOTH removals
        found_length = any(kw in contribution + key_math for kw in
                           ["length norm", "length normal", "per-response", "/|o|", "remove length",
                            "1/|o", "|oi|", "response length"])
        found_std   = any(kw in contribution + key_math for kw in
                          ["std norm", "std normal", "variance", "/σ", "remove std", "group std",
                           "std(", "standard deviation"])

        if found_length and found_std and mission_success:
            result["verdict"] = "PASS"
        elif found_length and found_std and not mission_success:
            result["verdict"] = "PARTIAL — extraction correct, code crashed (healer issue)"
        elif not found_length or not found_std:
            result["verdict"] = "FAIL — extraction missed key modification"
            if not found_length:
                result["notes"].append("❌ Did not identify removal of length normalization")
            if not found_std:
                result["notes"].append("❌ Did not identify removal of std normalization")
        else:
            result["verdict"] = "FAIL"

    elif test_num == 2:
        # DAPO: must get asymmetric clip + token-level loss; should flag dynamic sampling
        found_asymmetric = any(kw in contribution + key_math for kw in
                               ["asymmetric", "clip-higher", "ε_low", "epsilon_low", "0.2", "0.28"])
        found_token_level = any(kw in contribution + key_math for kw in
                                ["token-level", "token level", "per-token", "sequence-level", 
                                 "sum_{t=1", "sum over tokens", "\\sum_{t"])
        flagged_dynamic   = any(kw in contribution + key_math + feasible for kw in
                                ["dynamic sampling", "partial", "out-of-scope", "pre-step"])

        if found_asymmetric and found_token_level and mission_success:
            result["verdict"] = "PASS" if flagged_dynamic else "PASS (dynamic sampling not flagged — minor)"
        elif found_asymmetric and not found_token_level:
            result["verdict"] = "PARTIAL — only implemented Clip-Higher, missed token-level loss"
        elif not found_asymmetric:
            result["verdict"] = "FAIL — missed asymmetric clipping entirely"
        elif not mission_success:
            result["verdict"] = "PARTIAL — extraction correct but code crashed"
        else:
            result["verdict"] = "FAIL"

        if not flagged_dynamic:
            result["notes"].append("⚠️  Dynamic sampling not flagged as out-of-scope")

    elif test_num == 3:
        # Kimi K2.5: SHOULD have low confidence and abort, OR low confidence with no code execution
        if confidence <= 0.4 and not mission_success:
            result["verdict"] = "PASS — system correctly identified insufficient information and aborted"
        elif confidence <= 0.4 and mission_success:
            result["verdict"] = "PARTIAL — low confidence but generated working code (should have aborted earlier)"
        elif confidence > 0.6:
            result["verdict"] = "FAIL — HIGH CONFIDENCE HALLUCINATION (most dangerous failure mode)"
            result["notes"].append(f"🚨 Confidence {confidence} on a source with no algorithm specification")
            if mission_success:
                result["notes"].append("🚨 System generated and ran code for an unspecified algorithm")
        else:
            result["verdict"] = "PARTIAL — confidence in grey zone (0.4-0.6)"

        # If it actually tried to run code, that's also bad
        if mission_success:
            result["notes"].append("🚨 System generated and ran code for an unspecified algorithm")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # Parse CLI — allow running a single test
    run_single = None
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        run_single = int(sys.argv[idx + 1])

    if run_single and run_single not in TESTS:
        print(f"Unknown test number: {run_single}. Valid: {list(TESTS.keys())}")
        sys.exit(1)

    order = [run_single] if run_single else EXECUTION_ORDER

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("  TOMEA V2 STRESS TEST SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Running: {[TESTS[t]['name'] for t in order]}")
    print("=" * 70)

    results = []
    orchestrator = TomeaOrchestrator()

    for test_num in order:
        test_def = TESTS[test_num]

        print("\n" + "=" * 70)
        print(f"  TEST {test_num}: {test_def['name']}")
        print(f"  {test_def['description']}")
        print("=" * 70 + "\n")

        # --- Prep ---
        arxiv_id  = test_def["arxiv_id"]
        paper_text = test_def["paper_text"]
        dataset   = test_def["dataset"]

        # Test 3 special case: fetch blog text, no arxiv_id
        if test_num == 3:
            paper_text = load_kimi_blog_text(test_def["blog_url"])
            # Orchestrator uses arxiv_id for run_dir naming; use a stub
            arxiv_id = "kimi-k2.5-parl"

        # --- Run ---
        start = time.time()
        mission_success = await orchestrator.run_mission(
            arxiv_id=arxiv_id,
            paper_text=paper_text,
            dataset_name=dataset,
        )
        elapsed = time.time() - start

        # --- Analyze ---
        # Find the run dir that was just created (most recent under runs/)
        run_base = os.path.abspath("runs")
        prefix = str(arxiv_id) if arxiv_id else "kimi-k2.5-parl"
        run_dirs = sorted(
            [os.path.join(run_base, d) for d in os.listdir(run_base) if d.startswith(prefix)],
            key=os.path.getmtime
        )
        run_dir = run_dirs[-1] if run_dirs else run_base

        result = analyze_result(test_num, test_def, run_dir, mission_success)
        result["elapsed_seconds"] = round(elapsed, 1)
        result["run_dir"] = run_dir
        results.append(result)

        # Print immediate verdict
        print(f"\n   ⏱️  Elapsed: {result['elapsed_seconds']}s")
        print(f"   📊 Verdict: {result['verdict']}")
        for note in result["notes"]:
            print(f"      {note}")

        # If Test 4 (regression) fails, abort — everything else is meaningless
        if test_num == 4 and result["verdict"] == "FAIL":
            print("\n🛑 REGRESSION TEST FAILED. Aborting remaining tests.")
            print("   The V2 refactor broke something fundamental. Fix before stress testing.")
            break

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STRESS TEST SUMMARY")
    print("=" * 70)
    for r in results:
        status_icon = "✅" if "PASS" in r["verdict"] else ("⚠️ " if "PARTIAL" in r["verdict"] else "❌")
        print(f"  {status_icon} Test {r['test']:>2} | {r['name']:<40} | {r['verdict']}")
    print("=" * 70)

    # Save full results JSON
    report_path = os.path.join("runs", f"stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs("runs", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  📄 Full report saved: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())