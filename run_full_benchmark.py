"""Comprehensive Benchmark Runner for Prototype 3.

Runs full evaluation across all 15 benchmark scenarios, compares systems,
and exports benchmark metrics to benchmark_results.json.
"""

import json
import logging
from finance_ops.ledger.journal import LedgerRepository
from finance_ops.benchmark.runner import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def print_banner(title: str) -> None:
    w = 88
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def run_full_benchmark():
    print_banner("AI FINANCE CONTROLLER: PROTOTYPE 4 (ReAct) BENCHMARK RUNNER")
    print("[*] Architecture: Evidence-Grounded Autonomous Investigation Agent")
    print("[*] Gemini Vertex AI Native Function Calling + Deterministic Fallback Engine\n")

    cases_per_batch = 100
    print(f"[METRIC: Throughput] Processing single batch of {cases_per_batch} complex synthetic transactions...")
    
    # Initialize ledger repository
    ledger_repo = LedgerRepository()
    
    results = run_benchmark(
        seeds=[42],
        cases_per_seed=cases_per_batch,
        run_ablations=False
    )

    # We use Prototype3 for the main reporting
    p3_stats = results["systems"]["Prototype4_GeminiReAct"]
    rule_stats = results["systems"]["RuleMatcher"]

    # Calculate raw counts from the first seed
    exceptions = results.get("honest_exception_list", [])
    
    # Process journal entries from results
    for entry in results.get("journal_entries", []):
        ledger_repo.post(entry)
    
    # We must calculate TP, FP, FN, UNCERTAIN
    # Since metrics.py doesn't return raw counts in the summary, we can approximate or if metrics engine was modified:
    # We can calculate the match rate and counts from the honest exception list.
    # Actually, the runner summary doesn't expose raw counts directly, but we have recall and precision.
    # We can just output the metrics.
    
    print(f"\n====================================================================================================")
    print(f"{'System':<25}| {'Match F1':<11}| {'Triage F1':<11}| {'False Match %':<15}| {'Cause Diag %':<15}| {'Cost Utility'}")
    print(f"----------------------------------------------------------------------------------------------------")
    for sys_name in ["ExactMatcher", "RuleMatcher", "Prototype1_Hybrid", "Prototype4_GeminiReAct"]:
        stats = results["systems"][sys_name]
        f1 = f"{stats['match_f1_score']*100:.1f}%"
        triage_f1 = f"{stats['triage_f1_score']*100:.1f}%"
        fmr = f"{stats['false_match_rate']*100:.1f}%"
        cause_diag = f"{stats['cause_diagnosis_accuracy']*100:.1f}%"
        utility = f"${stats['cost_weighted_utility']:,.2f}"
        print(f"{sys_name:<25}| {f1:<11}| {triage_f1:<11}| {fmr:<15}| {cause_diag:<15}| {utility}")
    print(f"====================================================================================================")
    
    print(f"\n[METRIC: Throughput] Processing {cases_per_batch} cases at {p3_stats.get('throughput_cases_per_sec', 0):.2f} cases/sec (p95 latency: {p3_stats.get('p95_latency_ms', 0)} ms)")

    print(f"\nLEDGER & CASH POSITION:")
    tb = ledger_repo.trial_balance()
    cp = ledger_repo.generate_cash_position_report()
    fc = ledger_repo.generate_forward_forecast(days=30)
    
    print(f"  [+] Trial Balance: Debit = Credit = INR {tb['total_debits_paise']/100:,.2f} (Unbalanced: {tb['difference_paise']})")
    print(f"  [+] Total Liquidity: INR {cp['total_liquidity_inr']:,.2f}")
    print(f"      - Cash at Bank: INR {cp['cash_at_bank_inr']:,.2f}")
    print(f"      - Unmatched Suspense: INR {cp['unmatched_suspense_inr']:,.2f}")
    print(f"  [+] 30-Day Forward Forecast:")
    print(f"      - Expected Cash Inflow (75% clearance): INR {fc['expected_inflow_inr']:,.2f}")
    print(f"      - Write-off Risk (25% aging): INR {fc['write_off_risk_inr']:,.2f}")
    print("")

    print(f"\nEXCEPTIONS:")
    shown = 0
    # Deduplicate exceptions to prevent spam
    seen_cases = set()
    for e in exceptions:
        cid = e.get("case_id", "UNKNOWN")
        if cid in seen_cases:
            continue
        seen_cases.add(cid)
        shown += 1
        if shown > 5:
            break
        reason = e.get("reason", "UNKNOWN")
        expl = e.get("explanation", "").strip()
        print(f"  {cid} -> [{reason}] \"{expl}\"")

    if len(seen_cases) > 5:
        print(f"  ... and {len(seen_cases) - 5} more.")

    # Section 4: Export JSON
    output_path = "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        if 'journal_entries' in results:
            del results['journal_entries']
        json.dump(results, f, indent=2)
    print(f"\n[+] Full benchmark metrics successfully saved to {output_path}")


if __name__ == "__main__":
    run_full_benchmark()

