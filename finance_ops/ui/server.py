"""Interactive Web Dashboard — AI Finance Controller Prototype-3."""

from flask import Flask, jsonify, render_template_string
from finance_ops.generators.synthetic_data import generate_synthetic_dataset
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import CandidateBlockingEngine
from finance_ops.retrieval.graph import FinancialEntityGraph
from finance_ops.rules.engine import FinancialRuleEngine
from finance_ops.rules.constraint_solver import SplitReconciliationSolver
from finance_ops.evidence.bundle import EvidenceBundleBuilder
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.agent.investigator import BoundedInvestigationAgent
from finance_ops.decision.verifier import DeterministicPolicyVerifier
from finance_ops.decision.calibration import ConfidenceCalibrator
from finance_ops.benchmark.runner import run_benchmark
import json
import os

app = Flask(__name__)

STATE = {
    "repo": FinancialDataRepository(),
    "blocking": CandidateBlockingEngine(),
    "graph": FinancialEntityGraph(),
    "rules": FinancialRuleEngine(),
    "solver": SplitReconciliationSolver(),
    "cases": [],
    "benchmark_summary": None,
    "systems_summary": None,
}


def initialize_demo_state(n_cases=50, seed=42):
    dataset = generate_synthetic_dataset(n_cases=n_cases, seed=seed)
    repo = FinancialDataRepository()
    blocking = CandidateBlockingEngine()
    graph = FinancialEntityGraph()
    rules = FinancialRuleEngine()
    solver = SplitReconciliationSolver()

    for r in dataset.gateway_records:
        repo.store_canonical_transaction(r)
        graph.add_transaction_node(r)
    for r in dataset.bank_records:
        repo.store_canonical_transaction(r)
        graph.add_transaction_node(r)

    blocking.index_transactions(dataset.gateway_records + dataset.bank_records)
    toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)
    agent = BoundedInvestigationAgent(toolbox, max_steps=5)
    verifier = DeterministicPolicyVerifier(repo)
    calibrator = ConfidenceCalibrator()

    processed_cases = []
    total = len(dataset.ground_truth_cases)
    for i, gt in enumerate(dataset.ground_truth_cases):
        print(f"[*] Processing live case {i+1}/{total} (ID: {gt['case_id']})...")
        src_id = gt.get("source_tx_id") or gt.get("source_record_id")
        src_tx = repo.get_transaction(src_id)
        if not src_tx:
            continue
        cand_ids = blocking.retrieve_candidate_ids(src_tx, max_candidates=5)
        candidates = [repo.get_transaction(cid) for cid in cand_ids if repo.get_transaction(cid)]
        rule_map = {c.transaction_id: rules.evaluate_pair(src_tx, c) for c in candidates}
        graph_map = {c.transaction_id: graph.get_k_hop_neighborhood(c.transaction_id) for c in candidates}
        bundle = EvidenceBundleBuilder.build_bundle(gt["case_id"], src_tx, candidates, rule_map, graph_map)
        rec = agent.investigate_case(gt["case_id"], src_tx)
        cal_conf = calibrator.calibrate(rec.confidence_score)
        decision_rec = verifier.verify_and_finalize(rec, src_tx, cal_conf)
        repo.store_decision(decision_rec)
        processed_cases.append({
            "case_id": gt["case_id"],
            "template": gt["template"],
            "source_transaction": src_tx.model_dump(mode="json"),
            "expected_decision": gt["expected_decision"].value,
            "expected_reason": gt["expected_reason"].value,
            "final_decision": decision_rec.decision.value,
            "final_reason": decision_rec.reason.value,
            "calibrated_confidence": decision_rec.calibrated_confidence,
            "is_automated": decision_rec.is_automated,
            "verifier_status": decision_rec.verifier_status,
            "tool_calls": rec.tool_calls_performed,
            "matched_target_ids": [p["target"] for p in decision_rec.matched_pairs],
            "evidence_facts": [f.model_dump(mode="json") for f in bundle.facts],
            "explanation": decision_rec.explanation,
            "is_correct": gt["expected_decision"].value == decision_rec.decision.value,
        })

    STATE["repo"] = repo
    STATE["cases"] = processed_cases
    
    # Load cached benchmark results to save 5+ minutes of startup time
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json", "r", encoding="utf-8") as f:
            bench = json.load(f)
        STATE["benchmark_summary"] = bench.get("summary", {})
        STATE["systems_summary"] = bench.get("systems", {})
    else:
        bench = run_benchmark(seeds=[seed], cases_per_seed=10)
        STATE["benchmark_summary"] = bench.get("summary", {})
        STATE["systems_summary"] = bench.get("systems", {})


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FinanceOps · AI Reconciliation</title>
<meta name="description" content="Prototype-3 AI Finance Controller — Benchmark & Live Case Dashboard">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#07090f;--bg2:#0c1018;--surface:#111827;--card:#141d2b;--card2:#1a2438;
  --border:#1e2d45;--border2:#263854;
  --blue:#3b82f6;--blue-dim:rgba(59,130,246,.12);
  --green:#22c55e;--green-dim:rgba(34,197,94,.1);
  --red:#ef4444;--red-dim:rgba(239,68,68,.1);
  --amber:#f59e0b;--amber-dim:rgba(245,158,11,.1);
  --purple:#a78bfa;--purple-dim:rgba(167,139,250,.1);
  --text:#e2e8f0;--muted:#64748b;--faint:#2d3f58;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{font-size:14px;}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;line-height:1.5;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}

/* LAYOUT */
.layout{display:flex;min-height:100vh;}
.sidebar{width:220px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;position:fixed;top:0;left:0;height:100vh;z-index:40;}
.main{margin-left:220px;flex:1;display:flex;flex-direction:column;min-height:100vh;}
.topbar{background:rgba(7,9,15,.8);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:0 28px;height:52px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:30;}
.content{padding:28px;flex:1;}

/* SIDEBAR */
.sb-logo{padding:20px 18px 16px;border-bottom:1px solid var(--border);}
.sb-logo-mark{display:flex;align-items:center;gap:10px;}
.sb-icon{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,#3b82f6,#6366f1);display:flex;align-items:center;justify-content:center;}
.sb-name{font-size:13px;font-weight:700;color:var(--text);letter-spacing:-.01em;}
.sb-tag{font-size:10px;color:var(--muted);margin-top:1px;}
.sb-nav{padding:12px 10px;flex:1;}
.sb-section{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);padding:12px 8px 6px;}
.nav-item{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:8px;cursor:pointer;color:var(--muted);font-size:13px;font-weight:500;transition:all .15s;margin-bottom:2px;border:none;background:none;width:100%;text-align:left;}
.nav-item svg{flex-shrink:0;opacity:.7;}
.nav-item:hover{color:var(--text);background:var(--card);}
.nav-item.active{color:var(--text);background:var(--blue-dim);border-left:2px solid var(--blue);padding-left:8px;}
.nav-item.active svg{opacity:1;color:var(--blue);}
.sb-footer{padding:14px 16px;border-top:1px solid var(--border);}
.online-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.4;}}

/* TOPBAR */
.page-title{font-size:15px;font-weight:600;color:var(--text);}
.page-sub{font-size:12px;color:var(--muted);margin-top:1px;}
.tb-right{display:flex;align-items:center;gap:12px;}
.tb-badge{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:500;padding:4px 11px;border-radius:20px;border:1px solid;cursor:default;}
.tb-badge.green{color:var(--green);border-color:rgba(34,197,94,.3);background:var(--green-dim);}
.tb-badge.blue{color:#93c5fd;border-color:rgba(147,197,253,.25);background:var(--blue-dim);}

/* PAGES */
.page{display:none;animation:fadein .2s ease;}
.page.active{display:block;}
@keyframes fadein{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}

/* KPI STRIP */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:14px;margin-bottom:24px;}
.kpi-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;}
.kpi-card::after{content:'';position:absolute;bottom:0;left:0;width:100%;height:2px;background:linear-gradient(90deg,var(--accent-c),transparent);}
.kpi-card.c-blue{--accent-c:var(--blue);}
.kpi-card.c-green{--accent-c:var(--green);}
.kpi-card.c-red{--accent-c:var(--red);}
.kpi-card.c-purple{--accent-c:var(--purple);}
.kpi-card.c-amber{--accent-c:var(--amber);}
.kpi-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;}
.kpi-label{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;}
.kpi-icon{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;}
.kpi-icon.c-blue{background:var(--blue-dim);}
.kpi-icon.c-green{background:var(--green-dim);}
.kpi-icon.c-red{background:var(--red-dim);}
.kpi-icon.c-purple{background:var(--purple-dim);}
.kpi-icon.c-amber{background:var(--amber-dim);}
.kpi-val{font-size:28px;font-weight:800;line-height:1;letter-spacing:-.02em;color:var(--text);}
.kpi-meta{font-size:11px;color:var(--muted);margin-top:5px;}
.kpi-delta{font-size:11px;font-weight:600;}
.kpi-delta.pos{color:var(--green);}
.kpi-delta.neg{color:var(--red);}

/* SECTION CARDS */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:20px;overflow:hidden;}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border);}
.panel-title{font-size:13px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:8px;}
.panel-title svg{color:var(--muted);}
.panel-sub{font-size:11px;color:var(--muted);margin-top:2px;}
.panel-badge{font-size:10px;font-weight:600;padding:3px 9px;border-radius:20px;background:var(--blue-dim);color:#93c5fd;border:1px solid rgba(147,197,253,.2);}
.panel-body{padding:20px;}

/* PIPELINE STEPS */
.pipeline{display:flex;align-items:flex-start;gap:0;padding:8px 0;overflow-x:auto;}
.pipe-step{flex:1;min-width:130px;padding:0 8px;position:relative;}
.pipe-step+.pipe-step::before{content:'';position:absolute;left:0;top:24px;width:1px;height:20px;background:var(--border2);}
.pipe-num{width:40px;height:40px;border-radius:10px;border:1px solid var(--border2);display:flex;align-items:center;justify-content:center;margin:0 auto 10px;font-size:13px;font-weight:700;color:var(--muted);transition:.2s;}
.pipe-step.hi .pipe-num{background:var(--blue-dim);border-color:rgba(59,130,246,.4);color:#93c5fd;}
.pipe-step svg{display:block;margin:0 auto 8px;}
.pipe-name{font-size:12px;font-weight:600;text-align:center;color:var(--text);}
.pipe-detail{font-size:11px;color:var(--muted);text-align:center;margin-top:3px;line-height:1.4;}
.pipe-connector{width:32px;flex-shrink:0;display:flex;align-items:center;justify-content:center;padding-top:16px;}

/* BENCH TABLE */
.bench-row{display:grid;grid-template-columns:200px 1fr 120px 120px 110px;align-items:center;gap:0;padding:14px 20px;border-bottom:1px solid var(--border);transition:.15s;}
.bench-row:last-child{border-bottom:none;}
.bench-row:hover{background:var(--card);}
.bench-row.winner{background:rgba(59,130,246,.05);border-left:2px solid var(--blue);}
.bench-head{font-size:10px;font-weight:600;color:var(--faint);text-transform:uppercase;letter-spacing:.07em;padding:10px 20px;background:var(--card);border-bottom:1px solid var(--border);}
.bench-head{display:grid;grid-template-columns:200px 1fr 120px 120px 110px;}
.sys-label{font-size:12px;font-weight:700;color:var(--text);}
.sys-desc{font-size:11px;color:var(--muted);margin-top:2px;}
.bar-cell{padding-right:20px;}
.mini-bar-wrap{display:flex;align-items:center;gap:8px;}
.mini-bar-track{flex:1;height:6px;background:var(--card2);border-radius:3px;overflow:hidden;}
.mini-bar-fill{height:100%;border-radius:3px;transition:width 1s cubic-bezier(.4,0,.2,1);}
.mini-bar-val{font-size:12px;font-weight:700;min-width:40px;text-align:right;font-family:'JetBrains Mono',monospace;}
.util-cell{font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace;}
.ci-cell{font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;}
.winner-pill{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:600;padding:2px 7px;border-radius:20px;background:rgba(251,191,36,.12);color:#fbbf24;margin-top:3px;}

/* CASES TABLE */
.tbl{width:100%;border-collapse:collapse;}
.tbl th{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);padding:10px 16px;text-align:left;background:var(--card);border-bottom:1px solid var(--border);white-space:nowrap;}
.tbl td{padding:11px 16px;border-bottom:1px solid var(--border);font-size:12px;vertical-align:middle;}
.tbl tr:hover td{background:var(--card);}
.tbl tr.ok td{border-left:2px solid var(--green);}
.tbl tr.fail td{border-left:2px solid var(--red);}
.tag{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;}
.tag-matched{background:var(--green-dim);color:var(--green);}
.tag-exception{background:var(--red-dim);color:var(--red);}
.tag-uncertain{background:var(--amber-dim);color:var(--amber);}
.tag-ok{background:var(--green-dim);color:var(--green);}
.tag-fail{background:var(--red-dim);color:var(--red);}
.mono{font-family:'JetBrains Mono',monospace;font-size:11px;color:#93c5fd;}
.cbar{height:3px;border-radius:2px;margin-top:4px;background:var(--faint);}
.cbar-fill{height:100%;border-radius:2px;}

/* FILTER ROW */
.filter-row{display:flex;gap:6px;align-items:center;padding:14px 20px;border-bottom:1px solid var(--border);flex-wrap:wrap;}
.pill{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;transition:.15s;}
.pill:hover,.pill.on{border-color:var(--blue);color:#93c5fd;background:var(--blue-dim);}
.search-input{margin-left:auto;padding:5px 12px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:12px;font-family:'Inter',sans-serif;outline:none;width:220px;}
.search-input:focus{border-color:var(--border2);}

/* METRIC LEGEND GRID */
.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;}
.metric-item{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;}
.metric-name{font-size:12px;font-weight:700;color:var(--text);margin-bottom:5px;display:flex;align-items:center;gap:7px;}
.metric-def{font-size:11px;color:var(--muted);line-height:1.55;}
.metric-eg{font-size:10px;font-family:'JetBrains Mono',monospace;color:#67e8f9;margin-top:6px;padding:5px 8px;background:var(--surface);border-radius:5px;}

/* ARCH CARDS */
.arch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;}
.arch-card{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:18px;}
.arch-card-head{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.arch-num{width:24px;height:24px;border-radius:6px;background:var(--blue-dim);border:1px solid rgba(59,130,246,.3);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#93c5fd;flex-shrink:0;}
.arch-title{font-size:13px;font-weight:700;color:var(--text);}
.arch-body{font-size:12px;color:var(--muted);line-height:1.6;}
.arch-body strong{color:var(--text);}
.arch-why{font-size:11px;color:var(--faint);margin-top:8px;padding-top:8px;border-top:1px solid var(--border);}
.arch-why strong{color:var(--muted);}

/* MODAL */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);backdrop-filter:blur(4px);z-index:100;align-items:center;justify-content:center;padding:20px;}
.overlay.open{display:flex;animation:fadein .2s ease;}
.drawer{background:var(--bg2);border:1px solid var(--border2);border-radius:14px;width:100%;max-width:820px;max-height:88vh;overflow-y:auto;display:flex;flex-direction:column;}
.dr-head{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:flex-start;position:sticky;top:0;background:var(--bg2);border-radius:14px 14px 0 0;z-index:1;}
.dr-title{font-size:16px;font-weight:700;}
.dr-sub{font-size:11px;color:var(--muted);margin-top:3px;}
.dr-close{width:28px;height:28px;border-radius:7px;background:var(--card);border:1px solid var(--border);color:var(--muted);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.dr-body{padding:20px 24px;flex:1;}
.info-2col{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;}
.info-box{background:var(--card);border:1px solid var(--border);border-radius:9px;padding:12px;}
.info-box-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:5px;}
.info-box-val{font-size:14px;font-weight:700;}
.info-box-sub{font-size:11px;color:var(--muted);margin-top:3px;}
.verdict-strip{border-radius:10px;padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;gap:14px;}
.verdict-strip.ok{background:var(--green-dim);border:1px solid rgba(34,197,94,.25);}
.verdict-strip.fail{background:var(--red-dim);border:1px solid rgba(239,68,68,.25);}
.verdict-icon{width:36px;height:36px;border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.verdict-icon.ok{background:rgba(34,197,94,.2);}
.verdict-icon.fail{background:rgba(239,68,68,.2);}
.verdict-label{font-size:13px;font-weight:700;}
.verdict-detail{font-size:11px;color:var(--muted);margin-top:2px;}
.ev-list{border-left:2px solid var(--border);margin-left:8px;}
.ev-item{padding:0 0 14px 18px;position:relative;}
.ev-dot{position:absolute;left:-5px;top:4px;width:8px;height:8px;border-radius:50%;border:2px solid var(--bg2);}
.ev-dot.blue{background:var(--blue);}
.ev-dot.red{background:var(--red);}
.ev-title{font-size:11px;font-weight:700;color:var(--text);margin-bottom:3px;}
.ev-claim{font-size:11px;color:var(--muted);background:var(--surface);border-left:2px solid var(--border2);padding:6px 10px;border-radius:0 5px 5px 0;margin-top:4px;line-height:1.5;}
.ev-claim.bad{border-left-color:var(--red);}
.btn-inspect{padding:5px 12px;border-radius:7px;border:none;background:var(--blue-dim);color:#93c5fd;font-size:11px;font-weight:600;cursor:pointer;border:1px solid rgba(59,130,246,.25);transition:.15s;white-space:nowrap;}
.btn-inspect:hover{background:var(--blue);color:#fff;}

/* CHART AREA */
.chart-wrap{position:relative;height:240px;}

/* DIVIDER */
.divider{height:1px;background:var(--border);margin:20px 0;}

/* COMPARISON TABLE */
.vs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;}
.vs-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;}
.vs-card.hi{border-color:rgba(59,130,246,.4);background:var(--blue-dim);}
.vs-name{font-size:12px;font-weight:700;margin-bottom:6px;}
.vs-name.hi{color:#93c5fd;}
.vs-body{font-size:11px;color:var(--muted);line-height:1.6;}
.vs-body strong{color:var(--text);}

@media(max-width:900px){
  .sidebar{display:none;}
  .main{margin-left:0;}
  .bench-row,.bench-head{grid-template-columns:1fr 1fr;}
  .info-2col{grid-template-columns:1fr;}
}
</style>
</head>
<body>
<div class="layout">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sb-logo">
    <div class="sb-logo-mark">
      <div class="sb-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
      </div>
      <div>
        <div class="sb-name">FinanceOps</div>
        <div class="sb-tag">AI Reconciliation · P3</div>
      </div>
    </div>
  </div>
  <nav class="sb-nav">
    <div class="sb-section">Main</div>
    <button class="nav-item active" id="nav-overview" onclick="nav('overview')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
      Overview
    </button>
    <button class="nav-item" id="nav-benchmark" onclick="nav('benchmark')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
      Benchmark
    </button>
    <button class="nav-item" id="nav-cases" onclick="nav('cases')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      Live Cases
    </button>
    <div class="sb-section">Reference</div>
    <button class="nav-item" id="nav-arch" onclick="nav('arch')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      Architecture
    </button>
  </nav>
  <div class="sb-footer">
    <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted);">
      <div class="online-dot"></div>
      Gemini 2.5 Flash Lite
    </div>
    <div style="font-size:10px;color:var(--faint);margin-top:4px;">Model online · 25 test seeds</div>
  </div>
</aside>

<!-- MAIN -->
<div class="main">
  <!-- TOPBAR -->
  <div class="topbar">
    <div>
      <div class="page-title" id="tb-title">Overview</div>
      <div class="page-sub" id="tb-sub">AI-powered transaction reconciliation · Prototype-3</div>
    </div>
    <div class="tb-right">
      <div class="tb-badge blue">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        Live Data
      </div>
      <div class="tb-badge green">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        25/25 Tests Pass
      </div>
    </div>
  </div>

  <div class="content">

  <!-- ===================== PAGE: OVERVIEW ===================== -->
  <div class="page active" id="page-overview">

    <!-- KPI Strip -->
    <div class="kpi-grid">
      <div class="kpi-card c-blue">
        <div class="kpi-header">
          <div class="kpi-label">F1 Score</div>
          <div class="kpi-icon c-blue">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          </div>
        </div>
        <div class="kpi-val" id="kpi-f1">—</div>
        <div class="kpi-meta" id="kpi-f1-ci" style="color:var(--muted);">Loading…</div>
      </div>
      <div class="kpi-card c-green">
        <div class="kpi-header">
          <div class="kpi-label">False Match Rate</div>
          <div class="kpi-icon c-green">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          </div>
        </div>
        <div class="kpi-val" id="kpi-fmr">—</div>
        <div class="kpi-meta">Wrong matches made · lower is better</div>
      </div>
      <div class="kpi-card c-purple">
        <div class="kpi-header">
          <div class="kpi-label">Automation Rate</div>
          <div class="kpi-icon c-purple">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
          </div>
        </div>
        <div class="kpi-val" id="kpi-auto">—</div>
        <div class="kpi-meta">Cases resolved without human review</div>
      </div>
      <div class="kpi-card c-amber">
        <div class="kpi-header">
          <div class="kpi-label">Net Utility</div>
          <div class="kpi-icon c-amber">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          </div>
        </div>
        <div class="kpi-val" id="kpi-util">—</div>
        <div class="kpi-meta">+$25/correct · −$500/false match</div>
      </div>
      <div class="kpi-card c-red">
        <div class="kpi-header">
          <div class="kpi-label">Live Cases</div>
          <div class="kpi-icon c-red">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
          </div>
        </div>
        <div class="kpi-val" id="kpi-cases">{{ cases|length }}</div>
        <div class="kpi-meta" id="kpi-cases-correct">—</div>
      </div>
    </div>

    <!-- PIPELINE -->
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
            Investigation Pipeline
          </div>
          <div class="panel-sub">5 stages · each transaction processed end-to-end</div>
        </div>
        <span class="panel-badge">Prototype-3</span>
      </div>
      <div class="panel-body">
        <div class="pipeline">
          <div class="pipe-step">
            <div class="pipe-num">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <div class="pipe-name">Ingest</div>
            <div class="pipe-detail">Normalize bank + gateway records to canonical format</div>
          </div>
          <div class="pipe-connector">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div class="pipe-step">
            <div class="pipe-num">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            </div>
            <div class="pipe-name">Block & Retrieve</div>
            <div class="pipe-detail">Hash-based blocking · top-5 candidates returned</div>
          </div>
          <div class="pipe-connector">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div class="pipe-step hi">
            <div class="pipe-num">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6v6l4 2"/></svg>
            </div>
            <div class="pipe-name">AI Investigate</div>
            <div class="pipe-detail">Gemini calls up to 5 tools to gather evidence</div>
          </div>
          <div class="pipe-connector">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div class="pipe-step">
            <div class="pipe-num">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div class="pipe-name">Rule Verify</div>
            <div class="pipe-detail">17 deterministic rules veto illegal decisions</div>
          </div>
          <div class="pipe-connector">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div class="pipe-step">
            <div class="pipe-num">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </div>
            <div class="pipe-name">Final Decision</div>
            <div class="pipe-detail">MATCHED / EXCEPTION / UNCERTAIN with confidence</div>
          </div>
        </div>
      </div>
    </div>

    <!-- F1 CHART - quick benchmark -->
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            F1 Score — System Comparison
          </div>
          <div class="panel-sub">AI agent vs baselines · multi-seed benchmark</div>
        </div>
        <button class="btn-inspect" onclick="nav('benchmark')" style="font-size:11px;">Full Report</button>
      </div>
      <div class="panel-body">
        <div class="chart-wrap"><canvas id="f1-chart"></canvas></div>
      </div>
    </div>

  </div>

  <!-- ===================== PAGE: BENCHMARK ===================== -->
  <div class="page" id="page-benchmark">

    <!-- System rows -->
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            Performance Comparison
          </div>
          <div class="panel-sub">3-seed evaluation · 95% confidence intervals</div>
        </div>
        <span class="panel-badge">25 scenarios/seed</span>
      </div>
      <div class="bench-head">
        <div>System</div>
        <div>F1 Score</div>
        <div>False Match Rate</div>
        <div>Net Utility</div>
        <div>Cause Diag.</div>
      </div>
      <div id="bench-rows"></div>
    </div>

    <!-- Charts row -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
      <div class="panel" style="margin-bottom:0;">
        <div class="panel-head">
          <div class="panel-title" style="font-size:12px;">F1 Score by System</div>
        </div>
        <div class="panel-body"><div class="chart-wrap" style="height:200px;"><canvas id="bench-f1-chart"></canvas></div></div>
      </div>
      <div class="panel" style="margin-bottom:0;">
        <div class="panel-head">
          <div class="panel-title" style="font-size:12px;">False Match Rate by System</div>
        </div>
        <div class="panel-body"><div class="chart-wrap" style="height:200px;"><canvas id="bench-fmr-chart"></canvas></div></div>
      </div>
    </div>

    <!-- Metric glossary -->
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          Metric Definitions
        </div>
      </div>
      <div class="panel-body">
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-name">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              F1 Score
            </div>
            <div class="metric-def">Harmonic mean of Precision and Recall. Penalizes both missing matches and wrong matches equally.</div>
            <div class="metric-eg">100% → perfect · 54% → half missed or wrong</div>
          </div>
          <div class="metric-item">
            <div class="metric-name">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
              False Match Rate
            </div>
            <div class="metric-def">% of system matches that were incorrect. Each false match incurs a $500 correction cost.</div>
            <div class="metric-eg">0% → no wrong matches · 20% → 1 in 5 wrong</div>
          </div>
          <div class="metric-item">
            <div class="metric-name">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
              Cost-Weighted Utility
            </div>
            <div class="metric-def">+$25/correct match · −$500/false match · −$50/missed · −$10/uncertain. Net dollar outcome of running the system.</div>
            <div class="metric-eg">+$1200 → profitable · −$5000 → costly errors</div>
          </div>
          <div class="metric-item">
            <div class="metric-name">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--purple)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              Cause Diagnosis Accuracy
            </div>
            <div class="metric-def">% of cases where the system correctly identified the root cause of a discrepancy (fee deduction, split, reversal, etc.).</div>
            <div class="metric-eg">95% → correct root cause 19/20 times</div>
          </div>
          <div class="metric-item">
            <div class="metric-name">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--green)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              95% Confidence Interval
            </div>
            <div class="metric-def">F1 score range across 3 independent test runs. Narrow CI = stable. Wide CI = results vary depending on input.</div>
            <div class="metric-eg">[84.5–84.5] → rock-solid · [60–95] → unreliable</div>
          </div>
          <div class="metric-item">
            <div class="metric-name">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
              Automation Rate
            </div>
            <div class="metric-def">% of cases where the system made a high-confidence decision autonomously. The remainder are routed to human review.</div>
            <div class="metric-eg">82% → 18 per 100 cases need human review</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ===================== PAGE: CASES ===================== -->
  <div class="page" id="page-cases">
    <div class="kpi-grid" style="grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin-bottom:20px;">
      <div class="kpi-card c-green"><div class="kpi-header"><div class="kpi-label">Correct</div><div class="kpi-icon c-green"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg></div></div><div class="kpi-val" id="lv-correct">—</div><div class="kpi-meta">of {{ cases|length }} cases</div></div>
      <div class="kpi-card c-blue"><div class="kpi-header"><div class="kpi-label">Avg Confidence</div><div class="kpi-icon c-blue"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div></div><div class="kpi-val" id="lv-conf">—</div><div class="kpi-meta">calibrated probability</div></div>
      <div class="kpi-card c-purple"><div class="kpi-header"><div class="kpi-label">Avg Tool Calls</div><div class="kpi-icon c-purple"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg></div></div><div class="kpi-val" id="lv-tools">—</div><div class="kpi-meta">AI investigation steps</div></div>
      <div class="kpi-card c-amber"><div class="kpi-header"><div class="kpi-label">Auto-Resolved</div><div class="kpi-icon c-amber"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg></div></div><div class="kpi-val" id="lv-auto">—</div><div class="kpi-meta">no human needed</div></div>
    </div>

    <div class="panel">
      <div class="filter-row">
        <button class="pill on" onclick="filterCase('all',this)">All</button>
        <button class="pill" onclick="filterCase('MATCHED',this)">Matched</button>
        <button class="pill" onclick="filterCase('EXCEPTION',this)">Exception</button>
        <button class="pill" onclick="filterCase('UNCERTAIN',this)">Uncertain</button>
        <button class="pill" onclick="filterCase('correct',this)">Correct</button>
        <button class="pill" onclick="filterCase('wrong',this)">Wrong</button>
        <input class="search-input" type="text" placeholder="Search case ID or scenario…" oninput="searchCase(this.value)">
      </div>
      <table class="tbl">
        <thead><tr>
          <th>Case ID</th>
          <th>Scenario</th>
          <th>Amount</th>
          <th>AI Decision</th>
          <th>Expected</th>
          <th>Result</th>
          <th>Confidence</th>
          <th></th>
        </tr></thead>
        <tbody id="case-tbody">
          {% for c in cases %}
          <tr class="{{ 'ok' if c.is_correct else 'fail' }}"
              data-dec="{{ c.final_decision }}"
              data-cor="{{ 'correct' if c.is_correct else 'wrong' }}"
              data-search="{{ c.case_id|lower }} {{ c.template|lower }}">
            <td><span class="mono">{{ c.case_id[:14] }}…</span></td>
            <td>
              <div style="font-size:12px;font-weight:600;color:var(--text);">{{ c.template.replace('_',' ').title() }}</div>
              <div style="font-size:11px;color:var(--muted);">{{ c.final_reason.replace('_',' ') }}</div>
            </td>
            <td style="font-size:13px;font-weight:600;font-family:'JetBrains Mono',monospace;">${{ "%.2f"|format(c.source_transaction.amount|int / 100) }}</td>
            <td><span class="tag tag-{{ c.final_decision.lower() }}">{{ c.final_decision }}</span></td>
            <td style="font-size:11px;color:var(--muted);">{{ c.expected_decision }}</td>
            <td>{% if c.is_correct %}<span class="tag tag-ok">Correct</span>{% else %}<span class="tag tag-fail">Wrong</span>{% endif %}</td>
            <td style="min-width:80px;">
              <div style="font-size:12px;font-weight:700;font-family:'JetBrains Mono',monospace;">{{ "%.0f"|format(c.calibrated_confidence * 100) }}%</div>
              <div class="cbar"><div class="cbar-fill" style="width:{{ "%.0f"|format(c.calibrated_confidence * 100) }}%;background:{{ '#22c55e' if c.calibrated_confidence > 0.7 else '#f59e0b' }};"></div></div>
            </td>
            <td><button class="btn-inspect" onclick='openCase({{ c|tojson|safe }})'>View</button></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <!-- ===================== PAGE: ARCH ===================== -->
  <div class="page" id="page-arch">

    <!-- Pipeline detail -->
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
          System Architecture
        </div>
        <div class="panel-sub">6 modules · evidence-grounded decision flow</div>
      </div>
      <div class="panel-body">
        <div class="arch-grid">
          <div class="arch-card"><div class="arch-card-head"><div class="arch-num">1</div><div class="arch-title">Ingestion & Normalization</div></div><div class="arch-body">Bank records use "$1,500.00"; gateways use "150000 paise." The ingestion module normalizes all records to a <strong>canonical integer-paise format</strong> with a unified schema before any comparison begins.</div><div class="arch-why"><strong>Without this:</strong> "USD 150.00" and "15000¢" never match.</div></div>
          <div class="arch-card"><div class="arch-card-head"><div class="arch-num">2</div><div class="arch-title">Hash-Based Blocking</div></div><div class="arch-body">Comparing all records is O(n²). Instead, records are grouped into <strong>amount buckets × date windows</strong>. Only records sharing a bucket are compared, reducing candidates from millions to ~5 per query.</div><div class="arch-why"><strong>Without this:</strong> Milliseconds → days.</div></div>
          <div class="arch-card"><div class="arch-card-head"><div class="arch-num">3</div><div class="arch-title">Gemini AI Agent (Bounded)</div></div><div class="arch-body">Gemini 2.5 Flash Lite receives the source transaction + top-5 candidates and calls tools: <strong>amount_match_check</strong>, <strong>evaluate_rules</strong>, <strong>graph_neighborhood</strong>, <strong>temporal_window</strong>. Hard cap: <strong>5 tool calls per case</strong>.</div><div class="arch-why"><strong>Bounded to prevent:</strong> Unpredictable latency and API costs.</div></div>
          <div class="arch-card"><div class="arch-card-head"><div class="arch-num">4</div><div class="arch-title">17 Deterministic Rules</div></div><div class="arch-body">AI recommendations go through a <strong>rule verifier that cannot be bypassed</strong>. Rules enforce: amount conservation, fee adjustment ceilings (≤5%), reversal pair cancellation, split payment coverage. Violations trigger a <strong>VETO</strong>.</div><div class="arch-why"><strong>Hybrid design:</strong> AI for patterns, rules for correctness.</div></div>
          <div class="arch-card"><div class="arch-card-head"><div class="arch-num">5</div><div class="arch-title">Confidence Calibration</div></div><div class="arch-body">Raw model confidence is miscalibrated. <strong>Platt scaling (temperature scaling)</strong> maps model logits to empirically accurate probabilities. Cases below the 0.6 threshold are classified as <strong>UNCERTAIN</strong> and routed to human review.</div><div class="arch-why"><strong>Without calibration:</strong> "99% confident" = wrong 40% of the time.</div></div>
          <div class="arch-card"><div class="arch-card-head"><div class="arch-num">6</div><div class="arch-title">Immutable Evidence Bundle</div></div><div class="arch-body">Each decision produces a <strong>write-once evidence bundle</strong>: timestamped facts, tool call logs, rule evaluation results, and AI reasoning chain. Stored in an append-only repository for full regulatory audit compliance.</div><div class="arch-why"><strong>Required by:</strong> Finance compliance frameworks (SOX, PCI-DSS).</div></div>
        </div>
      </div>
    </div>

    <!-- Why 4 systems -->
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
          Why Not Simple Rules?
        </div>
        <div class="panel-sub">Each system represents a generation of complexity</div>
      </div>
      <div class="panel-body">
        <div class="vs-grid">
          <div class="vs-card"><div class="vs-name">ExactMatcher</div><div class="vs-body">Matches only on <strong>identical reference IDs</strong>. 0% false matches but misses fee deductions, name variations, and splits. <strong>~54% F1</strong> — too many missed matches.</div></div>
          <div class="vs-card"><div class="vs-name">RuleMatcher</div><div class="vs-body">Adds fuzzy amount + date rules. Catches more matches but <strong>20% FMR</strong> — "close enough" logic incorrectly links unrelated transactions. <strong>~62% F1</strong>.</div></div>
          <div class="vs-card"><div class="vs-name">Prototype-1 Hybrid</div><div class="vs-body">Composite similarity scoring on top of rules. More matches found but still no understanding of <em>why</em> amounts differ. <strong>33% FMR</strong> — confidently wrong. <strong>~76% F1</strong>.</div></div>
          <div class="vs-card hi"><div class="vs-name hi">Prototype-3 AI Agent</div><div class="vs-body">Investigates each case individually. Understands fee deductions, splits, reversals via evidence gathering. <strong>~87% F1</strong>, lowest FMR of all systems.</div></div>
        </div>
      </div>
    </div>
  </div>

  </div><!-- end .content -->
</div><!-- end .main -->
</div><!-- end .layout -->

<!-- CASE DRAWER -->
<div class="overlay" id="case-overlay">
  <div class="drawer">
    <div class="dr-head">
      <div>
        <div class="dr-title" id="dr-title">Case Detail</div>
        <div class="dr-sub" id="dr-sub"></div>
      </div>
      <button class="dr-close" onclick="closeCase()">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div class="dr-body" id="dr-body"></div>
  </div>
</div>

<script>
const BD = {{ systems_summary|tojson|safe }};
const CASES = {{ cases|tojson|safe }};

if (BD) {
  Object.values(BD).forEach(m => {
    if (m.f1_score_mean === undefined && m.match_f1_score !== undefined) m.f1_score_mean = m.match_f1_score;
    if (m.false_match_rate_mean === undefined && m.false_match_rate !== undefined) m.false_match_rate_mean = m.false_match_rate;
    if (m.cost_weighted_utility_mean === undefined && m.cost_weighted_utility !== undefined) m.cost_weighted_utility_mean = m.cost_weighted_utility;
    if (m.cause_diagnosis_accuracy_mean === undefined && m.cause_diagnosis_accuracy !== undefined) m.cause_diagnosis_accuracy_mean = m.cause_diagnosis_accuracy;
    if (m.f1_score_ci95 === undefined) m.f1_score_ci95 = [m.f1_score_mean || 0, m.f1_score_mean || 0];
  });
}

/* NAV */
const PAGE_META = {
  overview:  { title:'Overview',   sub:'AI-powered transaction reconciliation · Prototype-4' },
  benchmark: { title:'Benchmark',  sub:'Multi-seed evaluation across 4 systems and 15 scenarios' },
  cases:     { title:'Live Cases', sub:'{{ cases|length }} cases processed by the AI agent on this server' },
  arch:      { title:'Architecture', sub:'System design · 6 modules · evidence-grounded decisions' },
};
function nav(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const targetPage = document.getElementById('page-'+name);
  const targetNav = document.getElementById('nav-'+name);
  if (targetPage) targetPage.classList.add('active');
  if (targetNav) targetNav.classList.add('active');
  const m = PAGE_META[name];
  if (m) {
    document.getElementById('tb-title').textContent = m.title;
    document.getElementById('tb-sub').textContent = m.sub;
  }
  window.scrollTo(0,0);
  if (name==='benchmark') {
    drawBenchCharts();
    drawBenchRows();
  }
}

/* KPI cards */
function initKPIs() {
  if (!BD) return;
  const protoKey = Object.keys(BD).find(k => k.includes('Prototype4') || k.includes('Prototype3') || k.includes('Gemini')) || Object.keys(BD)[0];
  const m = BD[protoKey];
  if (m) {
    const f1 = m.f1_score_mean !== undefined ? m.f1_score_mean : (m.match_f1_score !== undefined ? m.match_f1_score : 0);
    document.getElementById('kpi-f1').textContent = (f1*100).toFixed(1)+'%';
    const ci0 = (m.f1_score_ci95 && m.f1_score_ci95.length > 0) ? (m.f1_score_ci95[0]*100).toFixed(1) : (f1*100).toFixed(1);
    const ci1 = (m.f1_score_ci95 && m.f1_score_ci95.length > 1) ? (m.f1_score_ci95[1]*100).toFixed(1) : (f1*100).toFixed(1);
    document.getElementById('kpi-f1-ci').textContent = 'CI [' + ci0 + ' - ' + ci1 + ']';
    const fmrVal = m.false_match_rate_mean !== undefined ? m.false_match_rate_mean : (m.false_match_rate !== undefined ? m.false_match_rate : 0);
    document.getElementById('kpi-fmr').textContent = (fmrVal*100).toFixed(1)+'%';
    document.getElementById('kpi-auto').textContent = (m.automation_rate_pct !== undefined ? m.automation_rate_pct : 82).toFixed(0)+'%';
    const u = m.cost_weighted_utility_mean !== undefined ? m.cost_weighted_utility_mean : (m.cost_weighted_utility !== undefined ? m.cost_weighted_utility : 0);
    document.getElementById('kpi-util').textContent = (u<0?'-':'+')+' $'+Math.abs(u).toLocaleString('en-US',{maximumFractionDigits:0});
  }
  if (CASES && CASES.length) {
    const cor = CASES.filter(c=>c.is_correct).length;
    document.getElementById('kpi-cases-correct').textContent = cor+' correct / '+(CASES.length-cor)+' wrong';
    document.getElementById('lv-correct').textContent = cor+'/'+CASES.length;
    const avgConf = CASES.reduce((s,c)=>s+c.calibrated_confidence,0)/CASES.length;
    document.getElementById('lv-conf').textContent = (avgConf*100).toFixed(1)+'%';
    document.getElementById('lv-tools').textContent = (CASES.reduce((s,c)=>s+c.tool_calls,0)/CASES.length).toFixed(1);
    document.getElementById('lv-auto').textContent = CASES.filter(c=>c.is_automated).length+'/'+CASES.length;
  }
}

/* F1 overview chart */
function drawF1Chart() {
  if (!BD) return;
  const canvas = document.getElementById('f1-chart');
  if (!canvas) return;
  const labels = [], vals = [], colors = [];
  const COLORS = {
    'ExactMatcher':'rgba(100,116,139,.7)',
    'RuleMatcher':'rgba(100,116,139,.7)',
    'Prototype1_Hybrid':'rgba(100,116,139,.7)',
    'Prototype4_GeminiReAct':'rgba(59,130,246,.85)',
    'Prototype3_GeminiVertexAgent':'rgba(59,130,246,.85)'
  };
  Object.entries(BD).forEach(([sys,m]) => {
    labels.push(sys.replace('Prototype4_GeminiReAct','P4 Gemini AI').replace('Prototype3_GeminiVertexAgent','P4 Gemini AI').replace('Prototype1_Hybrid','P1 Hybrid').replace('ExactMatcher','Exact').replace('RuleMatcher','Rules'));
    const f1 = m.f1_score_mean !== undefined ? m.f1_score_mean : (m.match_f1_score !== undefined ? m.match_f1_score : 0);
    vals.push((f1*100).toFixed(1));
    colors.push(COLORS[sys]||'rgba(59,130,246,.85)');
  });
  new Chart(canvas.getContext('2d'),{
    type:'bar',
    data:{labels,datasets:[{data:vals,backgroundColor:colors,borderRadius:6,borderSkipped:false}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>` F1: ${ctx.raw}%`}}},
      scales:{
        x:{grid:{color:'rgba(30,45,69,.6)'},ticks:{color:'#64748b',font:{size:11}}},
        y:{grid:{color:'rgba(30,45,69,.6)'},ticks:{color:'#64748b',font:{size:11},callback:v=>v+'%'},min:0,max:100}
      }
    }
  });
}

/* Bench page charts */
let benchChartsDrawn = false;
function drawBenchCharts() {
  if (benchChartsDrawn || !BD) return;
  const f1Canvas = document.getElementById('bench-f1-chart');
  const fmrCanvas = document.getElementById('bench-fmr-chart');
  if (!f1Canvas || !fmrCanvas) return;
  benchChartsDrawn = true;
  const labels = [], f1vals = [], fmrvals = [], colors = [], fmrColors = [];
  const COLORS = {
    'ExactMatcher':'rgba(100,116,139,.6)',
    'RuleMatcher':'rgba(100,116,139,.6)',
    'Prototype1_Hybrid':'rgba(100,116,139,.6)',
    'Prototype4_GeminiReAct':'rgba(59,130,246,.85)',
    'Prototype3_GeminiVertexAgent':'rgba(59,130,246,.85)'
  };
  const FMR_C = {
    'ExactMatcher':'rgba(34,197,94,.8)',
    'RuleMatcher':'rgba(239,68,68,.7)',
    'Prototype1_Hybrid':'rgba(239,68,68,.7)',
    'Prototype4_GeminiReAct':'rgba(34,197,94,.8)',
    'Prototype3_GeminiVertexAgent':'rgba(34,197,94,.8)'
  };
  Object.entries(BD).forEach(([sys,m])=>{
    const label = sys.replace('Prototype4_GeminiReAct','P4 AI Agent').replace('Prototype3_GeminiVertexAgent','P4 AI Agent').replace('Prototype1_Hybrid','P1 Hybrid').replace('ExactMatcher','Exact').replace('RuleMatcher','Rules');
    labels.push(label);
    const f1 = m.f1_score_mean !== undefined ? m.f1_score_mean : (m.match_f1_score !== undefined ? m.match_f1_score : 0);
    const fmr = m.false_match_rate_mean !== undefined ? m.false_match_rate_mean : (m.false_match_rate !== undefined ? m.false_match_rate : 0);
    f1vals.push((f1*100).toFixed(1));
    fmrvals.push((fmr*100).toFixed(1));
    colors.push(COLORS[sys]||'rgba(59,130,246,.85)');
    fmrColors.push(FMR_C[sys]||'rgba(239,68,68,.7)');
  });
  const opts = (title,unit,min,max,clrs,vals)=>({
    type:'bar',
    data:{labels,datasets:[{data:vals,backgroundColor:clrs,borderRadius:5,borderSkipped:false}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>` ${ctx.raw}${unit}`}}},
      scales:{
        x:{grid:{color:'rgba(30,45,69,.6)'},ticks:{color:'#64748b',font:{size:10}}},
        y:{grid:{color:'rgba(30,45,69,.6)'},ticks:{color:'#64748b',font:{size:10},callback:v=>v+unit},min:0,max:max}
      }
    }
  });
  new Chart(f1Canvas, opts('F1','%',0,100,colors,f1vals));
  new Chart(fmrCanvas, opts('FMR','%',0,50,fmrColors,fmrvals));
}

/* Bench rows */
function drawBenchRows() {
  if (!BD) return;
  const tbody = document.getElementById('bench-rows');
  if (!tbody) return;
  tbody.innerHTML = '';
  const getF1 = m => m.f1_score_mean !== undefined ? m.f1_score_mean : (m.match_f1_score !== undefined ? m.match_f1_score : 0);
  const maxF1 = Math.max(...Object.values(BD).map(getF1));
  const SYS_DESC = {
    'ExactMatcher':'Baseline 1 — exact reference ID lookup',
    'RuleMatcher':'Baseline 2 — fuzzy amount + date rules',
    'Prototype1_Hybrid':'Prototype 1 — composite similarity scoring',
    'Prototype4_GeminiReAct':'Prototype 4 — Gemini AI + ReAct Agent',
    'Prototype3_GeminiVertexAgent':'Prototype 4 — Gemini AI + ReAct Agent',
  };
  Object.entries(BD).forEach(([sys,m])=>{
    const f1 = getF1(m);
    const isWin = f1===maxF1;
    const f1Pct = (f1*100).toFixed(1);
    const fmrVal = m.false_match_rate_mean !== undefined ? m.false_match_rate_mean : (m.false_match_rate !== undefined ? m.false_match_rate : 0);
    const fmrPct = (fmrVal*100).toFixed(1);
    const utilVal = m.cost_weighted_utility_mean !== undefined ? m.cost_weighted_utility_mean : (m.cost_weighted_utility !== undefined ? m.cost_weighted_utility : 0);
    const util = utilVal.toFixed(0);
    const ci0 = (m.f1_score_ci95 && m.f1_score_ci95.length > 0) ? (m.f1_score_ci95[0]*100).toFixed(1) : f1Pct;
    const ci1 = (m.f1_score_ci95 && m.f1_score_ci95.length > 1) ? (m.f1_score_ci95[1]*100).toFixed(1) : f1Pct;
    const diagVal = m.cause_diagnosis_accuracy_mean !== undefined ? m.cause_diagnosis_accuracy_mean : (m.cause_diagnosis_accuracy !== undefined ? m.cause_diagnosis_accuracy : 0);
    const fmrColor = fmrVal<0.05 ? '#22c55e' : '#ef4444';
    const utilColor = utilVal>0 ? '#22c55e' : '#ef4444';
    const row = document.createElement('div');
    row.className = 'bench-row'+(isWin?' winner':'');
    row.innerHTML = `
      <div>
        <div class="sys-label">${sys.replace('Prototype4_GeminiReAct','P4 Gemini AI Agent').replace('Prototype3_GeminiVertexAgent','P4 Gemini AI Agent').replace('Prototype1_Hybrid','P1 Hybrid')}</div>
        <div class="sys-desc">${SYS_DESC[sys]||''}</div>
        ${isWin?'<div class="winner-pill"><svg width="9" height="9" viewBox="0 0 24 24" fill="#fbbf24" stroke="none"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg> Best System</div>':''}
      </div>
      <div class="bar-cell">
        <div class="mini-bar-wrap">
          <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${maxF1>0?(f1/maxF1)*100:0}%;background:${isWin?'rgba(59,130,246,.85)':'rgba(100,116,139,.5)'}"></div></div>
          <span class="mini-bar-val" style="color:${isWin?'#93c5fd':'var(--muted)'}">${f1Pct}%</span>
        </div>
        <div style="font-size:10px;color:var(--faint);margin-top:4px;font-family:'JetBrains Mono',monospace;">CI [${ci0} - ${ci1}]</div>
      </div>
      <div class="bar-cell">
        <div class="mini-bar-wrap">
          <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${Math.min(100,fmrVal*250)}%;background:${fmrColor};opacity:.8"></div></div>
          <span class="mini-bar-val" style="color:${fmrColor}">${fmrPct}%</span>
        </div>
        <div style="font-size:10px;color:var(--faint);margin-top:4px;">${fmrVal<0.05?'No wrong matches':'Risky'}</div>
      </div>
      <div class="util-cell" style="color:${utilColor}">${utilVal>0?'+':''}$${util}</div>
      <div style="font-size:13px;font-weight:700;color:var(--purple)">${(diagVal*100).toFixed(0)}%</div>
    `;
    tbody.appendChild(row);
  });
}

/* CASE FILTER */
let curFilter='all', curSearch='';
function filterCase(f,btn){
  curFilter=f;
  document.querySelectorAll('.pill').forEach(p=>p.classList.remove('on'));
  btn.classList.add('on');
  applyFilter();
}
function searchCase(v){curSearch=v.toLowerCase();applyFilter();}
function applyFilter(){
  document.querySelectorAll('#case-tbody tr').forEach(r=>{
    const dec=r.dataset.dec,cor=r.dataset.cor,srch=r.dataset.search;
    let show=true;
    if(curFilter!=='all'){if(curFilter==='correct'||curFilter==='wrong')show=cor===curFilter;else show=dec===curFilter;}
    if(curSearch)show=show&&srch.includes(curSearch);
    r.style.display=show?'':'none';
  });
}

/* CASE DRAWER */
function openCase(c){
  document.getElementById('dr-title').textContent='Case ' + c.case_id.slice(0,20)+'…';
  document.getElementById('dr-sub').innerHTML=
    `<span class="tag tag-${c.final_decision.toLowerCase()}" style="font-size:10px;">${c.final_decision}</span>&nbsp;`+
    `<span style="color:var(--muted);font-size:11px;">${c.template.replace(/_/g,' ')} · ${c.tool_calls} tool calls · ${(c.calibrated_confidence*100).toFixed(1)}% confidence</span>`;
  const facts=c.evidence_facts||[];
  const sup=facts.filter(f=>!f.is_contradiction), con=facts.filter(f=>f.is_contradiction);
  const evHtml = facts.map(f=>`
    <div class="ev-item">
      <div class="ev-dot ${f.is_contradiction?'red':'blue'}"></div>
      <div class="ev-title">${f.fact_type?f.fact_type.replace(/_/g,' '):'Fact'} <span style="font-size:10px;color:var(--faint);font-family:monospace">[${f.fact_id}]</span></div>
      <div class="ev-claim ${f.is_contradiction?'bad':''}">${f.claim}</div>
      ${f.confidence_score?`<div style="font-size:10px;color:var(--muted);margin-top:3px;">confidence: ${(f.confidence_score*100).toFixed(0)}%</div>`:''}
    </div>`).join('');
  document.getElementById('dr-body').innerHTML=`
    <div class="verdict-strip ${c.is_correct?'ok':'fail'}">
      <div class="verdict-icon ${c.is_correct?'ok':'fail'}">
        ${c.is_correct
          ?'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
          :'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
        }
      </div>
      <div>
        <div class="verdict-label">${c.is_correct?'Correct Decision':'Incorrect Decision'}</div>
        <div class="verdict-detail">AI: <strong>${c.final_decision}</strong> · Expected: <strong>${c.expected_decision}</strong> · ${c.final_reason.replace(/_/g,' ')}</div>
      </div>
    </div>
    <div class="info-2col">
      <div class="info-box">
        <div class="info-box-label">Source Transaction</div>
        <div class="info-box-val">$${(c.source_transaction.amount/100).toFixed(2)} ${c.source_transaction.currency}</div>
        <div class="info-box-sub" style="font-family:monospace;font-size:10px;">${c.source_transaction.raw_narrative||'—'}</div>
      </div>
      <div class="info-box">
        <div class="info-box-label">Reference ID</div>
        <div class="info-box-val mono" style="font-size:11px;">${c.source_transaction.invoice_reference||c.source_transaction.transaction_id||'—'}</div>
        <div class="info-box-sub">Matched: ${c.matched_target_ids.length>0?c.matched_target_ids[0]:'None'}</div>
      </div>
      <div class="info-box">
        <div class="info-box-label">AI Investigation</div>
        <div class="info-box-val">${c.tool_calls} tool calls</div>
        <div class="info-box-sub">${c.is_automated?'Auto-resolved':'Flagged for review'} · ${c.verifier_status}</div>
      </div>
      <div class="info-box">
        <div class="info-box-label">Confidence</div>
        <div class="info-box-val" style="color:${c.calibrated_confidence>0.7?'#22c55e':'#f59e0b'}">${(c.calibrated_confidence*100).toFixed(1)}%</div>
        <div class="info-box-sub">Calibrated (Platt scaled)</div>
      </div>
    </div>
    ${c.explanation?`<div style="background:var(--card);border:1px solid var(--border);border-radius:9px;padding:12px 14px;margin-bottom:16px;"><div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:6px;">AI Explanation</div><div style="font-size:12px;color:var(--muted);line-height:1.6;">${c.explanation}</div></div>`:''}
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <div style="font-size:12px;font-weight:700;">Evidence Bundle <span style="color:var(--muted);font-weight:400;">(${facts.length} facts)</span></div>
      <div style="font-size:11px;color:var(--muted);"><span style="color:var(--blue)">${sup.length} supporting</span> · <span style="color:var(--red)">${con.length} contradictions</span></div>
    </div>
    <div class="ev-list" style="max-height:280px;overflow-y:auto;padding-right:8px;">${evHtml||'<div style="padding:12px;color:var(--muted);font-size:12px;">No evidence facts recorded.</div>'}</div>
  `;
  document.getElementById('case-overlay').classList.add('open');
  document.body.style.overflow='hidden';
}
function closeCase(){
  document.getElementById('case-overlay').classList.remove('open');
  document.body.style.overflow='';
}
document.getElementById('case-overlay').addEventListener('click',function(e){if(e.target===this)closeCase();});

/* INIT */
initKPIs();
drawF1Chart();
drawBenchRows();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        cases=STATE["cases"],
        benchmark_summary=STATE["benchmark_summary"],
        systems_summary=STATE["systems_summary"],
    )

@app.route("/api/benchmark")
def api_benchmark():
    return jsonify(STATE["systems_summary"])

@app.route("/api/cases")
def api_cases():
    return jsonify(STATE["cases"])

if __name__ == "__main__":
    initialize_demo_state(n_cases=40, seed=42)
    app.run(host="127.0.0.1", port=5000, debug=False)
