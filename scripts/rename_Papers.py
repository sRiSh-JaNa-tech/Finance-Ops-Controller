"""Rename research PDFs using their embedded or first-page titles.

Progress is saved after every file in ``.pdf_rename_state.json`` so the
operation can be resumed safely after an interruption or individual failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_FILENAME = ".pdf_rename_state.json"
INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE = re.compile(r"\s+")
MAX_FILENAME_LENGTH = 180


def load_pdf_reader() -> tuple[Any, str]:
	"""Load an installed PDF reader, preferring PyMuPDF's text extraction."""
	try:
		import fitz

		return fitz, "fitz"
	except ImportError:
		try:
			from PyPDF2 import PdfReader

			return PdfReader, "pypdf2"
		except ImportError as exc:
			raise RuntimeError(
				"Install a PDF library first: python -m pip install pymupdf"
			) from exc


def clean_title(title: str) -> str:
	"""Convert extracted title text into a valid, readable Windows filename."""
	title = unicodedata.normalize("NFKC", title)
	title = title.replace("\u00ad", "")
	title = WHITESPACE.sub(" ", title).strip(" .")
	title = INVALID_WINDOWS_CHARS.sub("-", title)
	title = re.sub(r"-{2,}", "-", title).strip(" .")
	if title.upper() in {"CON", "PRN", "AUX", "NUL"}:
		title = f"{title}-paper"
	for prefix in ("COM", "LPT"):
		if title.upper().startswith(prefix) and title[len(prefix):].isdigit():
			title = f"{title}-paper"
	return title[:MAX_FILENAME_LENGTH].rstrip(" .")


def looks_like_title(line: str) -> bool:
	line = WHITESPACE.sub(" ", line).strip()
	lowered = line.lower()
	if len(line) < 8 or len(line) > 240:
		return False
	if lowered.startswith(("http://", "https://", "www.", "doi:")):
		return False
	if re.fullmatch(r"[\d\W_]+", line):
		return False
	if "journal homepage" in lowered or "copyright" in lowered:
		return False
	return sum(character.isalpha() for character in line) >= 5


def first_page_title(text: str) -> str | None:
	lines = [WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
	lines = [line for line in lines if line]
	candidates: list[str] = []
	for line in lines[:35]:
		if not looks_like_title(line):
			continue
		if re.search(r"\b(authors?|abstract|keywords?|arxiv)\b", line, re.I):
			continue
		candidates.append(line)
	if not candidates:
		return None
	return max(candidates[:8], key=len)


def first_page_title_from_layout(page: Any) -> str | None:
	"""Choose the largest meaningful text block near the top of a page."""
	candidates: list[tuple[float, float, str]] = []
	for block in page.get_text("dict").get("blocks", []):
		if "lines" not in block:
			continue
		lines = block["lines"]
		for line_index, line in enumerate(lines):
			text = " ".join(
				span["text"].strip() for span in line["spans"] if span["text"].strip()
			)
			if not looks_like_title(text):
				continue
			if re.search(
				r"\b(authors?|abstract|keywords?|article info|copyright|received|accepted|"
				r"university|department|repository|journal|issn|license)\b",
				text,
				re.I,
			):
				continue
			if text.isupper() and len(text) <= 20:
				continue
			if re.match(r"^\d+[\s,.)-]", text):
				continue
			font_size = max(
				(span["size"] for span in line["spans"] if span["text"].strip()),
				default=0.0,
			)
			title_lines = [text]
			for continuation in lines[line_index + 1 :]:
				continuation_text = " ".join(
					span["text"].strip()
					for span in continuation["spans"]
					if span["text"].strip()
				)
				continuation_size = max(
					(
						span["size"]
						for span in continuation["spans"]
						if span["text"].strip()
					),
					default=0.0,
				)
				if (
					continuation_size < font_size * 0.9
					or not looks_like_title(continuation_text)
					or re.search(
						r"\b(authors?|abstract|keywords?|university|department|repository)\b",
						continuation_text,
						re.I,
					)
				):
					break
				title_lines.append(continuation_text)
			text = " ".join(title_lines)
			candidates.append((font_size, -line["bbox"][1], text))
	if not candidates:
		return None
	return max(candidates)[2]


def extract_title(path: Path, reader_api: Any, reader_kind: str) -> str:
	if reader_kind == "fitz":
		document = reader_api.open(path)
		try:
			metadata_title = (document.metadata or {}).get("title", "")
			if metadata_title and looks_like_title(metadata_title):
				return metadata_title
			if document.page_count:
				title = first_page_title_from_layout(document[0])
				if title:
					return title
				text = document[0].get_text("text")
			else:
				text = ""
		finally:
			document.close()
	else:
		reader = reader_api(str(path))
		metadata = reader.metadata or {}
		metadata_title = str(metadata.get("/Title") or "").strip()
		if metadata_title and looks_like_title(metadata_title):
			return metadata_title
		text = reader.pages[0].extract_text() if reader.pages else ""

	title = first_page_title(text or "")
	if not title:
		raise ValueError("could not find a title in PDF metadata or first page")
	return title


def load_state(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {"version": 1, "files": {}}
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
		if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
			raise ValueError("state file has an invalid structure")
		return data
	except (OSError, json.JSONDecodeError, ValueError) as exc:
		raise RuntimeError(f"Cannot read {path}: {exc}") from exc


def save_state(path: Path, state: dict[str, Any]) -> None:
	temporary_path = path.with_suffix(path.suffix + ".tmp")
	temporary_path.write_text(
		json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	temporary_path.replace(path)


def unique_target(directory: Path, title: str, source: Path, reserved: set[str]) -> Path:
	base = clean_title(title) or source.stem
	candidate = directory / f"{base}.pdf"
	number = 2
	while (
		candidate.name.casefold() in reserved
		and candidate.name.casefold() != source.name.casefold()
	) or (
		candidate.exists() and candidate.resolve() != source.resolve()
	):
		candidate = directory / f"{base} ({number}).pdf"
		number += 1
	reserved.add(candidate.name.casefold())
	return candidate


def fingerprint(path: Path) -> dict[str, int]:
	stat = path.stat()
	return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def process(directory: Path, dry_run: bool, retry_failed: bool, reset: bool) -> int:
	state_path = directory.parent / STATE_FILENAME
	state = {"version": 1, "files": {}} if reset else load_state(state_path)
	files = state["files"]
	reader_api, reader_kind = load_pdf_reader()
	reserved = {path.name.casefold() for path in directory.glob("*.pdf")}
	failures = 0

	for source in sorted(directory.glob("*.pdf"), key=lambda item: item.name.casefold()):
		key = source.name
		record = files.get(key, {})
		current_fingerprint = fingerprint(source)
		if (
			record.get("status") == "renamed"
			and record.get("fingerprint") == current_fingerprint
		):
			print(f"SKIP  {source.name} (already completed)")
			continue
		if record.get("status") == "failed" and not retry_failed:
			print(f"SKIP  {source.name} (failed previously; use --retry-failed)")
			failures += 1
			continue

		try:
			title = extract_title(source, reader_api, reader_kind)
			target = unique_target(directory, title, source, reserved)
			record.update(
				{
					"status": "planned" if dry_run else "renamed",
					"fingerprint": current_fingerprint,
					"title": title,
					"target": target.name,
					"updated_at": datetime.now(timezone.utc).isoformat(),
				}
			)
			if dry_run:
				print(f"DRY   {source.name} -> {target.name}")
			else:
				source.rename(target)
				print(f"OK    {source.name} -> {target.name}")
		except Exception as exc:  # Continue and checkpoint even if one PDF is bad.
			record.update(
				{
					"status": "failed",
					"fingerprint": current_fingerprint,
					"error": f"{type(exc).__name__}: {exc}",
					"updated_at": datetime.now(timezone.utc).isoformat(),
				}
			)
			failures += 1
			print(f"FAIL  {source.name}: {exc}", file=sys.stderr)
		files[key] = record
		save_state(state_path, state)

	print(f"State saved to {state_path}")
	return 1 if failures else 0


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dry-run", action="store_true", help="show planned names without renaming"
	)
	parser.add_argument(
		"--retry-failed", action="store_true", help="retry files recorded as failed"
	)
	parser.add_argument(
		"--reset", action="store_true", help="discard the checkpoint and start over"
	)
	args = parser.parse_args()
	directory = Path(__file__).resolve().parent / "research_papers"
	if not directory.is_dir():
		parser.error(f"PDF directory does not exist: {directory}")
	return process(directory, args.dry_run, args.retry_failed, args.reset)


if __name__ == "__main__":
	raise SystemExit(main())
