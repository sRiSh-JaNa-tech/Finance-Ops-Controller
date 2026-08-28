"""
Concurrency Benchmarking for Retrieve-Rank-Route-Reason Cascade Architecture.
Sweeps max_workers through 1, 2, 4, 8, 16 to find the throughput saturation point.
"""

import asyncio
import time
import numpy as np
from finance_ops.generators.synthetic_data import generate_synthetic_dataset
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import MultiPassBlockingEngine
from finance_ops.agent.cascade_router import CascadeReconciliationPipeline

async def run_concurrency_sweep():
    print("=" * 70)
    print("  ASYNC CONCURRENCY SWEEP: THROUGHPUT & LATENCY SCALING")
    print("=" * 70)
    
    # 1. Setup dataset (100 cases to keep benchmark fast, but realistic)
    ds = generate_synthetic_dataset(n_cases=100, seed=42)
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
    
    workers_list = [1, 2, 4, 8, 16]
    
    print(f"{'Workers':<10} | {'Throughput (c/s)':<20} | {'P95 Latency (ms)':<20}")
    print("-" * 55)
    
    for w in workers_list:
        start_t = time.perf_counter()
        
        results = await cascade.process_batch_async(
            cases=ds.ground_truth_cases,
            candidate_lookup=candidate_lookup,
            repo=repo,
            max_workers=w
        )
        
        end_t = time.perf_counter()
        total_sec = max(end_t - start_t, 0.001)
        
        throughput = len(ds.ground_truth_cases) / total_sec
        latencies = [r.get("latency_ms", 0.0) for r in results]
        p95 = np.percentile(latencies, 95) if latencies else 0.0
        
        print(f"{w:<10} | {throughput:<20.1f} | {p95:<20.1f}")
        
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_concurrency_sweep())
