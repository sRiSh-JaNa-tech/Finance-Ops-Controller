"""
Calibration script for routing thresholds.
Sweeps thresholds over a development dataset to identify the lowest threshold that guarantees
a False Match Rate (FMR) <= 0.5%.
"""

import asyncio
from finance_ops.generators.synthetic_data import generate_synthetic_dataset
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import MultiPassBlockingEngine
from finance_ops.agent.cascade_router import CascadeReconciliationPipeline
from finance_ops.benchmark.metrics import FinancialReconciliationMetrics

async def run_calibration():
    print("=" * 70)
    print("  THRESHOLD CALIBRATION: GUARANTEEING FMR <= 0.5%")
    print("=" * 70)
    
    # 1. Setup dev dataset (200 cases)
    ds = generate_synthetic_dataset(n_cases=200, seed=1234)
    repo = FinancialDataRepository()
    for r in ds.gateway_records + ds.bank_records:
        repo.store_canonical_transaction(r)
        
    blocking = MultiPassBlockingEngine()
    blocking_pairs = blocking.generate_candidate_pairs(ds.gateway_records, ds.bank_records)
    candidate_lookup = {}
    for src, tgt, keys in blocking_pairs:
        if src.transaction_id not in candidate_lookup:
            candidate_lookup[src.transaction_id] = []
        candidate_lookup[src.transaction_id].append(tgt)
            
    cascade = CascadeReconciliationPipeline(repository=repo, mode="offline")
    
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98]
    best_threshold = None
    
    print(f"{'Threshold':<15} | {'False Match Rate':<20} | {'Automation Rate':<20}")
    print("-" * 60)
    
    for t in thresholds:
        cascade.estimator.fast_path_threshold = t
        
        results = await cascade.process_batch_async(
            cases=ds.ground_truth_cases,
            candidate_lookup=candidate_lookup,
            repo=repo,
            max_workers=8
        )
        
        metrics = FinancialReconciliationMetrics().evaluate_predictions(results, ds.ground_truth_cases)
        fmr = metrics["false_match_rate"]
        auto_rate = metrics["automation_rate"]
        
        print(f"{t:<15} | {fmr*100:<19.2f}% | {auto_rate*100:<19.2f}%")
        
        if fmr <= 0.005 and best_threshold is None:
            best_threshold = t
            
    print("=" * 70)
    if best_threshold:
        print(f"CALIBRATION COMPLETE: Freezing fast_path_threshold at {best_threshold}")
    else:
        print("WARNING: Could not find a threshold with FMR <= 0.5% in the sweep.")

if __name__ == "__main__":
    asyncio.run(run_calibration())
