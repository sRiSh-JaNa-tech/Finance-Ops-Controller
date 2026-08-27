import os
import json
import logging
from dotenv import load_dotenv
from finance_ops.agent.vertex_client import GeminiVertexReconciliationClient
from finance_ops.benchmark.runner import _generate_cases
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import CandidateBlockingEngine
from finance_ops.retrieval.graph import FinancialEntityGraph
from finance_ops.rules.engine import FinancialRuleEngine
from finance_ops.rules.constraint_solver import SplitReconciliationSolver

logging.basicConfig(level=logging.INFO)
load_dotenv()

repo = FinancialDataRepository()
blocking = CandidateBlockingEngine()
graph = FinancialEntityGraph()
rules = FinancialRuleEngine()
solver = SplitReconciliationSolver()
toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)

cases = _generate_cases(42, 1) # Generate 1 case

for case in cases:
    source_tx = case.source_transaction
    target_txs = case.target_transactions
    
    # Need to load them into the repo
    repo.store_canonical_transaction(source_tx)
    for t in target_txs:
        repo.store_canonical_transaction(t)
    blocking.index_transactions([source_tx] + target_txs)
    
    client = GeminiVertexReconciliationClient()
    print("Investigating Case:", case.case_id)
    rec = client.investigate_case(case.case_id, source_tx, [], toolbox, max_steps=5)
    print("Recommendation:", rec.recommended_decision)
    print("Reason:", rec.primary_reason)
    break
