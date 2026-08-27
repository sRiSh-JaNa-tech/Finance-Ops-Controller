"""Tests for Phase 2 Synthetic Data Generator and Fault Injection."""

from finance_ops.generators.synthetic_data import generate_synthetic_dataset
from finance_ops.generators.fault_injection import ScenarioTemplate
from finance_ops.core.models import DecisionLabel


def test_synthetic_data_generation_reproducibility():
    ds1 = generate_synthetic_dataset(n_cases=30, seed=123)
    ds2 = generate_synthetic_dataset(n_cases=30, seed=123)
    
    assert len(ds1.gateway_records) == len(ds2.gateway_records)
    assert len(ds1.bank_records) == len(ds2.bank_records)
    assert len(ds1.ground_truth_cases) == 30
    assert ds1.gateway_records[0].amount == ds2.gateway_records[0].amount


def test_scenario_stratification():
    ds = generate_synthetic_dataset(n_cases=50, seed=42)
    templates = [c["template"] for c in ds.ground_truth_cases]
    
    assert ScenarioTemplate.CLEAN_MATCH.value in templates
    assert ScenarioTemplate.FEE_TAX_VARIANCE.value in templates
    assert ScenarioTemplate.DELAYED_SETTLEMENT.value in templates
    assert ScenarioTemplate.SPLIT_PAYMENT.value in templates
    assert ScenarioTemplate.AMBIGUOUS_CANDIDATE.value in templates
