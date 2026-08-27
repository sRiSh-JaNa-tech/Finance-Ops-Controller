"""Field-Level Lineage, Transformation Tracking, and Cryptographic Provenance."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional
from finance_ops.core.models import FieldProvenance, SourceSystem, RawSourceRecord


def compute_raw_hash(payload: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash for raw payload dictionary."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_field_provenance(
    source_system: SourceSystem,
    source_field_name: str,
    raw_value: Any,
    normalized_value: Any,
    transformation_applied: str,
    confidence: float = 1.0,
) -> FieldProvenance:
    """Constructs an immutable FieldProvenance record."""
    return FieldProvenance(
        source_system=source_system,
        source_field_name=source_field_name,
        raw_value=str(raw_value) if raw_value is not None else "",
        normalized_value=str(normalized_value) if normalized_value is not None else "",
        transformation_applied=transformation_applied,
        transformation_timestamp=datetime.utcnow(),
        confidence_score=confidence,
    )


def create_raw_record(
    raw_record_id: str,
    source_system: SourceSystem,
    source_file_or_endpoint: str,
    raw_payload: Dict[str, Any],
) -> RawSourceRecord:
    """Creates a raw immutable source record with cryptographic checksum."""
    return RawSourceRecord(
        raw_record_id=raw_record_id,
        source_system=source_system,
        source_file_or_endpoint=source_file_or_endpoint,
        raw_payload=raw_payload,
        ingestion_timestamp=datetime.utcnow(),
        raw_hash=compute_raw_hash(raw_payload),
    )
