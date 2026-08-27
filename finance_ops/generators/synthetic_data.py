"""Synthetic Multi-Source Financial Data Generator for 15 Prototype 3 Benchmark Scenarios.

Generates realistic heterogeneous transactions across Razorpay Gateway, Bank Statements,
ERP Invoices, and GST Portals with ground truth audit labels.
"""

from typing import List, Dict, Tuple, Any, Optional
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone
import random
import uuid

from finance_ops.core.models import (
    SourceSystem, TransactionType, PaymentMethod, TransactionStatus,
    DecisionLabel, ReasonCode, CanonicalTransaction, InvoiceRecord, SettlementBatch
)
from finance_ops.core.invariants import quantize_amount
from finance_ops.generators.fault_injection import ScenarioTemplate


INDIAN_MERCHANT_POOL = [
    {"id": "M_RP_101", "name": "Flipkart Internet Private Limited", "aliases": ["FLIPKART INTERNET", "FLIPKART INDIA", "FLIPKART PVT LTD", "FLIPKARTT IND"], "vpa": "flipkart@icici", "gstin": "29AABCU9603R1ZJ"},
    {"id": "M_RP_102", "name": "Amazon Seller Services Pvt Ltd", "aliases": ["AMAZON WEB SERVICES", "AMZN PAY INDIA", "AMAZON SELLER SVCS"], "vpa": "amazonpay@apl", "gstin": "27AAACA1234B1Z5"},
    {"id": "M_RP_103", "name": "Zomato Limited", "aliases": ["ZOMATO FOOD", "ZOMATO LTD", "ZOMATO RESTAURANTS"], "vpa": "zomato@hdfcbank", "gstin": "07AAACZ1122C1Z8"},
    {"id": "M_RP_104", "name": "Swiggy Bundl Technologies", "aliases": ["SWIGGY BUNDL", "BUNDL TECHNOLOGIES PVT", "SWIGGY BANGALORE"], "vpa": "swiggy@axisbank", "gstin": "29AABCB3344D1Z2"},
    {"id": "M_RP_105", "name": "Reliance Retail Limited", "aliases": ["JIO MART", "RELIANCE DIGITAL", "RELIANCE RETAIL LTD"], "vpa": "jiopay@rbi", "gstin": "27AAACR5566E1Z1"},
]

CUSTOMER_POOL = [
    {"id": "C_IND_501", "name": "Aarav Sharma", "vpa": "aarav.sharma@okhdfcbank"},
    {"id": "C_IND_502", "name": "Priya Patel", "vpa": "priya.patel@okaxis"},
    {"id": "C_IND_503", "name": "Rohan Verma", "vpa": "rohan.v@icici"},
    {"id": "C_IND_504", "name": "Sneha Reddy", "vpa": "sneha.reddy@sbi"},
    {"id": "C_IND_505", "name": "Vikram Malhotra", "vpa": "vikram.m@paytm"},
]


class SyntheticFinancialDataset:
    """Encapsulates generated multi-source financial records along with ground truth evaluation labels."""

    def __init__(
        self,
        bank_records: List[CanonicalTransaction],
        gateway_records: List[CanonicalTransaction],
        invoices: List[InvoiceRecord],
        settlement_batches: List[SettlementBatch],
        ground_truth_cases: List[Dict[str, Any]],
        scenario_counts: Dict[str, int]
    ):
        self.bank_records = bank_records
        self.gateway_records = gateway_records
        self.invoices = invoices
        self.settlement_batches = settlement_batches
        self.ground_truth_cases = ground_truth_cases
        self.scenario_counts = scenario_counts


def generate_synthetic_dataset(
    n_cases: int = 60,
    seed: int = 42,
    scenario_distribution: Optional[Dict[ScenarioTemplate, float]] = None
) -> SyntheticFinancialDataset:
    """
    Generates a stratified multi-source financial dataset across 15 benchmark scenarios
    with latent ground truth linkage.
    """
    random.seed(seed)
    
    all_scenarios = [
        ScenarioTemplate.S01_CLEAN_EXACT_MATCH,
        ScenarioTemplate.S02_FEE_ADJUSTED_MDR,
        ScenarioTemplate.S03_GST_DISCREPANCY,
        ScenarioTemplate.S04_SPLIT_PAYMENT,
        ScenarioTemplate.S05_VALID_REVERSAL,
        ScenarioTemplate.S06_EXPIRED_REVERSAL,
        ScenarioTemplate.S07_DUPLICATE_REVERSAL,
        ScenarioTemplate.S08_MERCHANT_NAME_TYPO,
        ScenarioTemplate.S09_FX_ROUNDING,
        ScenarioTemplate.S10_UNEXPLAINED_MISMATCH,
        ScenarioTemplate.S11_CARD_T2_SETTLEMENT,
        ScenarioTemplate.S12_HOLIDAY_SETTLEMENT,
        ScenarioTemplate.S13_MISSING_APPROVAL_TOKEN,
        ScenarioTemplate.S14_CANDIDATE_TIE_AMBIGUITY,
        ScenarioTemplate.S15_REPEATED_MICRO_CREDIT_LEAKAGE,
        ScenarioTemplate.S16_HIDDEN_COMBINED_MUTATION,
    ]

    bank_records: List[CanonicalTransaction] = []
    gateway_records: List[CanonicalTransaction] = []
    invoices: List[InvoiceRecord] = []
    settlement_batches: List[SettlementBatch] = []
    ground_truth_cases: List[Dict[str, Any]] = []
    scenario_counts: Dict[str, int] = {s.value: 0 for s in all_scenarios}

    base_time = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)

    for case_idx in range(n_cases):
        # Round-robin selection through all 15 scenarios to ensure high coverage
        scenario = all_scenarios[case_idx % len(all_scenarios)]
        scenario_counts[scenario.value] += 1

        gt_tx_id = f"GT_TX_{case_idx+1000:04d}"
        merchant = random.choice(INDIAN_MERCHANT_POOL)
        customer = random.choice(CUSTOMER_POOL)
        tx_time = base_time + timedelta(days=random.randint(0, 15), hours=random.randint(1, 8), minutes=random.randint(0, 59))
        epoch_ts = int(tx_time.timestamp())

        base_amount_paise = random.randint(100000, 500000)  # ₹1,000 to ₹5,000
        base_dec = Decimal(base_amount_paise) / Decimal(100)
        inv_number = f"INV-2026-{case_idx+100:04d}"
        order_id = f"ORD-{case_idx+5000:05d}"
        utr_num = f"{random.randint(400000000000, 499999999999)}"

        # Default source (Razorpay)
        gw_tx_id = f"TXN_RP_{case_idx+1000:04d}"
        bank_tx_id = f"TXN_BANK_{case_idx+1000:04d}"

        if scenario == ScenarioTemplate.S01_CLEAN_EXACT_MATCH:
            # S01: Clean Exact Match
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                transaction_timestamp=tx_time,
                merchant_id=merchant["id"],
                merchant_name=merchant["name"],
                merchant_name_norm=merchant["name"].lower(),
                order_id=order_id,
                invoice_reference=inv_number,
                utr=utr_num,
                payment_method=PaymentMethod.UPI,
                status=TransactionStatus.CAPTURED,
                transaction_type=TransactionType.PAYMENT,
                raw_narrative=f"Payment for {inv_number} to {merchant['name']}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 3600,
                transaction_timestamp=tx_time + timedelta(hours=1),
                merchant_id=merchant["id"],
                merchant_name=merchant["name"],
                merchant_name_norm=merchant["name"].lower(),
                order_id=order_id,
                invoice_reference=inv_number,
                utr=utr_num,
                payment_method=PaymentMethod.UPI,
                status=TransactionStatus.CAPTURED,
                transaction_type=TransactionType.PAYMENT,
                raw_narrative=f"UPI/{utr_num}/{customer['name']}/{inv_number}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.EXACT_IDENTIFIER_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S02_FEE_ADJUSTED_MDR:
            # S02: Fee-Adjusted Gateway MDR (2% fee)
            fee_paise = int(round(base_amount_paise * 0.02))
            net_paise = base_amount_paise - fee_paise
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                fee_paise=fee_paise,
                net_paise=net_paise,
                amount=base_dec,
                fee=Decimal(fee_paise)/Decimal(100),
                net_amount=Decimal(net_paise)/Decimal(100),
                currency="INR",
                txn_timestamp=epoch_ts,
                transaction_timestamp=tx_time,
                merchant_id=merchant["id"],
                merchant_name=merchant["name"],
                order_id=order_id,
                invoice_reference=inv_number,
                utr=utr_num,
                payment_method=PaymentMethod.CARD,
                status=TransactionStatus.CAPTURED,
                raw_narrative=f"Card Payment {order_id} 2% MDR fee deducted"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=net_paise,
                amount=Decimal(net_paise)/Decimal(100),
                currency="INR",
                txn_timestamp=epoch_ts + 86400,
                transaction_timestamp=tx_time + timedelta(days=1),
                merchant_id=merchant["id"],
                merchant_name=merchant["name"],
                order_id=order_id,
                payment_method=PaymentMethod.CARD,
                status=TransactionStatus.CAPTURED,
                raw_narrative=f"SETTLEMENT/RAZORPAY/NET/{order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.FEE_ADJUSTED_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S03_GST_DISCREPANCY:
            # S03: GST Rate Discrepancy (12% applied instead of 18%)
            fee_paise = int(round(base_amount_paise * 0.02))
            wrong_gst_paise = int(round(fee_paise * 0.12))  # Error! Should be 18%
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                fee_paise=fee_paise,
                gst_paise=wrong_gst_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                transaction_timestamp=tx_time,
                merchant_id=merchant["id"],
                merchant_name=merchant["name"],
                order_id=order_id,
                payment_method=PaymentMethod.CARD,
                raw_narrative=f"Payment with non-compliant GST {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise - fee_paise - wrong_gst_paise,
                amount=Decimal(base_amount_paise - fee_paise - wrong_gst_paise)/Decimal(100),
                currency="INR",
                txn_timestamp=epoch_ts + 86400,
                transaction_timestamp=tx_time + timedelta(days=1),
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"SETTLEMENT/{order_id}/GST_MISMATCH"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.EXCEPTION,
                "expected_reason": ReasonCode.GST_CALCULATION_ERROR,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S04_SPLIT_PAYMENT:
            # S04: 1-to-3 Split Payment
            part1 = base_amount_paise // 3
            part2 = base_amount_paise // 3
            part3 = base_amount_paise - part1 - part2

            inv_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.ERP,
                source_record_id=inv_number,
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                transaction_timestamp=tx_time,
                merchant_id=merchant["id"],
                order_id=order_id,
                invoice_reference=inv_number,
                is_split=True,
                raw_narrative=f"ERP Invoice {inv_number} split in 3 tranches"
            )
            split_txs = []
            for i, p_amt in enumerate([part1, part2, part3]):
                stx = CanonicalTransaction(
                    transaction_id=f"TXN_BANK_{case_idx+1000:04d}_P{i+1}",
                    source_system=SourceSystem.BANK,
                    source_record_id=f"bnk_split_{case_idx+1000:04d}_{i+1}",
                    ground_truth_tx_id=gt_tx_id,
                    amount_paise=p_amt,
                    amount=Decimal(p_amt)/Decimal(100),
                    currency="INR",
                    txn_timestamp=epoch_ts + (i+1)*3600,
                    transaction_timestamp=tx_time + timedelta(hours=i+1),
                    merchant_id=merchant["id"],
                    order_id=order_id,
                    invoice_reference=inv_number,
                    raw_narrative=f"Split Payment {i+1}/3 for {inv_number}"
                )
                split_txs.append(stx)
                bank_records.append(stx)

            gateway_records.append(inv_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [t.transaction_id for t in split_txs],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.SPLIT_PAYMENT_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S05_VALID_REVERSAL:
            # S05: Valid Reversal within 90 days with approval code
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"rfnd_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400 * 5,  # 5 days later
                transaction_timestamp=tx_time + timedelta(days=5),
                merchant_id=merchant["id"],
                order_id=order_id,
                approval_code=f"AUTH_{random.randint(100000, 999999)}",
                reason_code="CUSTOMER_RETURN",
                is_refund=True,
                is_reversal=True,
                parent_transaction_id=f"orig_pay_{case_idx+1000:04d}",
                raw_narrative=f"Authorized Refund for {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_rev_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400 * 5 + 3600,
                transaction_timestamp=tx_time + timedelta(days=5, hours=1),
                merchant_id=merchant["id"],
                order_id=order_id,
                approval_code=f"AUTH_{random.randint(100000, 999999)}",
                is_refund=True,
                is_reversal=True,
                raw_narrative=f"REFUND/REV/{order_id}/CUSTOMER_RETURN"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.REVERSAL_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S06_EXPIRED_REVERSAL:
            # S06: Expired Reversal >90 days (120 days post-txn)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"rfnd_exp_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400 * 120,  # 120 days post txn
                transaction_timestamp=tx_time + timedelta(days=120),
                merchant_id=merchant["id"],
                order_id=order_id,
                approval_code=f"AUTH_{random.randint(100000, 999999)}",
                is_refund=True,
                is_reversal=True,
                raw_narrative=f"Late refund request for {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_exp_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,  # Original transaction date
                transaction_timestamp=tx_time,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Original purchase {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.EXCEPTION,
                "expected_reason": ReasonCode.EXPIRED_REVERSAL,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S07_DUPLICATE_REVERSAL:
            # S07: Duplicate Reversal Fraud (AB-1 fail)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"rfnd_dup_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400 * 2,
                transaction_timestamp=tx_time + timedelta(days=2),
                merchant_id=merchant["id"],
                order_id=order_id,
                reason_code="DUPLICATE_REVERSAL",
                is_refund=True,
                is_reversal=True,
                approval_code=f"AUTH_{random.randint(100000, 999999)}",
                raw_narrative=f"Duplicate refund for {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_dup_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                is_refund=True,
                is_reversal=True,
                raw_narrative=f"Prior refunded txn {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.EXCEPTION,
                "expected_reason": ReasonCode.DUPLICATE_REVERSAL,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S08_MERCHANT_NAME_TYPO:
            # S08: Merchant Name Typo ("Flipkart Internet" vs "Flipkartt Ind")
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                merchant_name=merchant["name"],
                order_id=order_id,
                invoice_reference=inv_number,
                payment_method=PaymentMethod.UPI,
                raw_narrative=f"Payment to {merchant['name']}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 1800,
                merchant_id=merchant["id"],
                merchant_name=random.choice(merchant["aliases"]),
                order_id=order_id,
                invoice_reference=inv_number,
                payment_method=PaymentMethod.UPI,
                raw_narrative=f"UPI/TXN/{order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.EXACT_IDENTIFIER_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S09_FX_ROUNDING:
            # S09: FX / Rounding Proximity (within 50 paise)
            rounding_paise = 45  # ₹0.45 difference
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"FX payment {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise + rounding_paise,
                amount=Decimal(base_amount_paise + rounding_paise)/Decimal(100),
                currency="INR",
                txn_timestamp=epoch_ts + 3600,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"FX conversion bank {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.FUZZY_ENTITY_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S10_UNEXPLAINED_MISMATCH:
            # S10: Unexplained Amount Mismatch (>4% difference)
            mismatch_amt = int(round(base_amount_paise * 1.05))
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Payment {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=f"DIFFERENT_TX_{case_idx}",
                amount_paise=mismatch_amt,
                amount=Decimal(mismatch_amt)/Decimal(100),
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=f"ORD-DIFF-{case_idx}",
                raw_narrative=f"Unrelated payment {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.EXCEPTION,
                "expected_reason": ReasonCode.AMOUNT_MISMATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S11_CARD_T2_SETTLEMENT:
            # S11: Card Normal T+2 Settlement
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                payment_method=PaymentMethod.CARD,
                raw_narrative=f"Card payment {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400 * 2,  # T+2 days
                merchant_id=merchant["id"],
                order_id=order_id,
                payment_method=PaymentMethod.CARD,
                raw_narrative=f"Card settlement D+2 {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.TIMING_ALIGNED_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S12_HOLIDAY_SETTLEMENT:
            # S12: Bank Holiday Cascade (T+5 settlement with warning)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                payment_method=PaymentMethod.CARD,
                raw_narrative=f"Payment before Diwali holidays {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400 * 5,  # T+5 days
                merchant_id=merchant["id"],
                order_id=order_id,
                payment_method=PaymentMethod.CARD,
                raw_narrative=f"Settlement post-holiday {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.TIMING_ALIGNED_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S13_MISSING_APPROVAL_TOKEN:
            # S13: Missing Approval Token on Refund (AI-1 fail -> UNCERTAIN)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"rfnd_noauth_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts + 86400,
                merchant_id=merchant["id"],
                order_id=order_id,
                approval_code=None,  # Missing auth code!
                is_refund=True,
                is_reversal=True,
                raw_narrative=f"Refund request without approval code {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Payment for {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.UNCERTAIN,
                "expected_reason": ReasonCode.MISSING_AUTHORIZATION,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S14_CANDIDATE_TIE_AMBIGUITY:
            # S14: Candidate Tie Ambiguity (2 identical amounts in bank stream)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"pay_tie_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                raw_narrative=f"Payment {merchant['name']}"
            )
            bank_tx1 = CanonicalTransaction(
                transaction_id=f"TXN_BANK_TIE1_{case_idx+1000:04d}",
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_tie1_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                raw_narrative=f"Bank Txn A {merchant['name']}"
            )
            bank_tx2 = CanonicalTransaction(
                transaction_id=f"TXN_BANK_TIE2_{case_idx+1000:04d}",
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_tie2_{case_idx+1000:04d}",
                ground_truth_tx_id=f"GT_OTHER_{case_idx}",
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                raw_narrative=f"Bank Txn B {merchant['name']}"
            )
            gateway_records.append(gw_tx)
            bank_records.extend([bank_tx1, bank_tx2])
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx1.transaction_id, bank_tx2.transaction_id],
                "expected_decision": DecisionLabel.UNCERTAIN,
                "expected_reason": ReasonCode.AMBIGUOUS_CANDIDATES,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S15_REPEATED_MICRO_CREDIT_LEAKAGE:
            # S15: Repeated Micro-Credit Leakage (AB-2 fail)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"leak_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Excessive micro refund {order_id}"
            )
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_leak_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Micro credit {order_id}"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.EXCEPTION,
                "expected_reason": ReasonCode.REVENUE_LEAKAGE_DETECTED,
                "scenario": scenario.value,
                "template": scenario.value
            })

        elif scenario == ScenarioTemplate.S16_HIDDEN_COMBINED_MUTATION:
            # S16: Combined Fee Adjustment + Timestamp mismatch + Typo (Zero-shot generalization test)
            gw_tx = CanonicalTransaction(
                transaction_id=gw_tx_id,
                source_system=SourceSystem.RAZORPAY,
                source_record_id=f"gw_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=base_amount_paise,
                amount=base_dec,
                currency="INR",
                txn_timestamp=epoch_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Settlement for {merchant['name']}"
            )
            # Apply 2% fee deduction, shift timestamp by 3 days, add typo
            bank_amt = quantize_amount(base_dec * Decimal("0.98"))
            bank_paise = int(bank_amt * 100)
            shifted_ts = epoch_ts + (3 * 24 * 3600)
            
            bank_tx = CanonicalTransaction(
                transaction_id=bank_tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"bnk_{case_idx+1000:04d}",
                ground_truth_tx_id=gt_tx_id,
                amount_paise=bank_paise,
                amount=bank_amt,
                currency="INR",
                txn_timestamp=shifted_ts,
                merchant_id=merchant["id"],
                order_id=order_id,
                raw_narrative=f"Setlement {merchant['aliases'][0][:10]} (Fee deducted)"
            )
            gateway_records.append(gw_tx)
            bank_records.append(bank_tx)
            ground_truth_cases.append({
                "case_id": f"CASE_{gw_tx_id}",
                "source_tx_id": gw_tx_id,
                "candidate_tx_ids": [bank_tx_id],
                "expected_decision": DecisionLabel.MATCHED,
                "expected_reason": ReasonCode.FEE_ADJUSTED_MATCH,
                "scenario": scenario.value,
                "template": scenario.value
            })

    return SyntheticFinancialDataset(
        bank_records=bank_records,
        gateway_records=gateway_records,
        invoices=invoices,
        settlement_batches=settlement_batches,
        ground_truth_cases=ground_truth_cases,
        scenario_counts=scenario_counts
    )
