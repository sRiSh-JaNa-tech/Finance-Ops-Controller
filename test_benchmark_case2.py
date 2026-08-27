import os
import json
import logging
from dotenv import load_dotenv
from finance_ops.agent.vertex_client import GeminiVertexReconciliationClient
from finance_ops.generators.synthetic_data import generate_synthetic_dataset
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import MultiPassBlockingEngine
from finance_ops.retrieval.graph import FinancialEntityGraph
from finance_ops.rules.engine import DeterministicRuleEngine
from finance_ops.rules.constraint_solver import SplitReconciliationSolver

logging.basicConfig(level=logging.INFO)
load_dotenv()

repo = FinancialDataRepository()
blocking = MultiPassBlockingEngine()
graph = FinancialEntityGraph()
rules = DeterministicRuleEngine()
solver = SplitReconciliationSolver()
toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)

dataset = generate_synthetic_dataset(n_cases=1, seed=42)

for case in dataset.ground_truth_cases:
    src_id = case["source_tx_id"]
    src_tx = dataset.gateway_records[0] # Just grab the first one for testing
    target_txs = dataset.bank_records[:5]
    
    repo.store_canonical_transaction(src_tx)
    for t in target_txs:
        repo.store_canonical_transaction(t)
    
    client = GeminiVertexReconciliationClient()
    print("Investigating Case:", case["case_id"])
    rec = client.investigate_case(case["case_id"], src_tx, target_txs, toolbox, max_steps=5)
    print("Recommendation:", rec.recommended_decision)
    print("Reason:", rec.primary_reason)
    break
