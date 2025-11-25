"""
traffic_summary.py

A standalone utility to scan transportation engineering documents for traffic count and forecast
information and produce concise summaries grouped by Application ID.

Dependencies:
- pdfplumber
- python-docx
- pandas
- openpyxl
- xlrd

Usage:
    python traffic_summary.py <input_folder> <output_file>

Example:
    python traffic_summary.py ./input_docs traffic_summaries.md
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class TrafficContent:
    base_years: List[str] = field(default_factory=list)
    sample_counts: List[str] = field(default_factory=list)
    forecast_years: List[str] = field(default_factory=list)
    forecast_volumes: List[str] = field(default_factory=list)
    growth_rates: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    pages: List[str] = field(default_factory=list)
    sheets: List[str] = field(default_factory=list)

    def has_data(self) -> bool:
        return any(
            [
                self.base_years,
                self.sample_counts,
                self.forecast_years,
                self.forecast_volumes,
                self.growth_rates,
                self.assumptions,
            ]
        )


@dataclass
class FileSummary:
    application_id: str
    file_name: str
    content: TrafficContent


# -----------------------------
# Keyword configuration
# -----------------------------
VOLUME_KEYWORDS = [
    "traffic count",
    "volume",
    "aadt",
    "turning movement",
    "tmc",
    "intersection count",
    "peak hour",
    "am peak",
    "pm peak",
]
TURNING_KEYWORDS = ["left", "right", "thru", "through", "u-turn", "uturn"]
DIRECTION_KEYWORDS = ["northbound", "nb", "southbound", "sb", "eastbound", "eb", "westbound", "wb"]
FORECAST_KEYWORDS = [
    "forecast",
    "future year",
    "growth rate",
    "cagr",
    "k-factor",
    "growth factor",
]
ASSUMPTION_KEYWORDS = [
    "assume",
    "assumption",
    "based on",
    "growth",
    "land use",
    "development",
    "population",
    "economic",
]
YEAR_PATTERN = re.compile(r"\b(20[3-9][0-9])\b")
GROWTH_PATTERN = re.compile(r"\b([0-9]+\.?[0-9]*)%\b")
COUNT_PATTERN = re.compile(r"\b(\d{1,5})\b")


# -----------------------------
# File extraction helpers
# -----------------------------
def extract_text_pdf(path: Path) -> Tuple[str, List[str]]:
    pages_with_data: List[str] = []
    texts: List[str] = []
    try:
        import pdfplumber
    except ImportError:
        logging.warning("pdfplumber is not installed; skipping PDF extraction for %s", path)
        return "", pages_with_data

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    texts.append(text)
                    pages_with_data.append(str(page_num))
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to process PDF %s: %s", path, exc)
    return "\n".join(texts), pages_with_data


def extract_text_docx(path: Path) -> Tuple[str, List[str]]:
    pages: List[str] = []
    try:
        import docx
    except ImportError:
        logging.warning("python-docx is not installed; skipping DOCX extraction for %s", path)
        return "", pages

    try:
        document = docx.Document(path)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        # DOCX does not carry page numbers easily; treat as single page.
        if text.strip():
            pages.append("1")
        return text, pages
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to process DOCX %s: %s", path, exc)
        return "", pages


def extract_text_doc(path: Path) -> Tuple[str, List[str]]:
    try:
        # Attempt basic binary read; textract is optional due to heavy dependency.
        try:
            import textract  # type: ignore
        except ImportError:
            logging.warning("textract is not installed; skipping DOC extraction for %s", path)
            return "", []
        text = textract.process(str(path)).decode("utf-8", errors="ignore")
        return text, ["1"] if text.strip() else []
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to process DOC %s: %s", path, exc)
        return "", []


def extract_text_excel(path: Path) -> Tuple[str, List[str]]:
    all_text: List[str] = []
    sheets: List[str] = []
    try:
        excel = pd.ExcelFile(path)
    except Exception as exc:
        logging.warning("Unable to open Excel file %s: %s", path, exc)
        return "", sheets

    for sheet_name in excel.sheet_names:
        try:
            df = excel.parse(sheet_name=sheet_name, dtype=str, header=None)
            flattened = "\n".join(
                f"{col}: {val}" for col, val in zip(df.columns, df.astype(str).fillna("").agg(" ".join, axis=1))
            )
            all_text.append(f"Sheet {sheet_name}\n{flattened}")
            sheets.append(sheet_name)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to read sheet %s in %s: %s", sheet_name, path, exc)
    return "\n".join(all_text), sheets


def extract_text_csv(path: Path) -> Tuple[str, List[str]]:
    sheet_name = path.name
    try:
        df = pd.read_csv(path, dtype=str, header=None, engine="python")
    except Exception as exc:
        logging.warning("Unable to read CSV file %s: %s", path, exc)
        return "", []
    flattened = "\n".join(
        f"{col}: {val}" for col, val in zip(df.columns, df.astype(str).fillna("").agg(" ".join, axis=1))
    )
    return f"Sheet {sheet_name}\n{flattened}", [sheet_name]


# -----------------------------
# Detection logic
# -----------------------------
def detect_traffic_content(text: str) -> TrafficContent:
    lowered = text.lower()
    content = TrafficContent()

    def find_keywords(keywords: Sequence[str]) -> bool:
        return any(keyword in lowered for keyword in keywords)

    if find_keywords(VOLUME_KEYWORDS):
        base_years = sorted(set(YEAR_PATTERN.findall(text)))
        content.base_years.extend(base_years)
        samples = _extract_sample_counts(text)
        content.sample_counts.extend(samples)

    if find_keywords(FORECAST_KEYWORDS):
        forecast_years = sorted(set(YEAR_PATTERN.findall(text)))
        content.forecast_years.extend(forecast_years)
        if forecast_years:
            forecast_samples = _extract_forecast_volumes(text, forecast_years)
            content.forecast_volumes.extend(forecast_samples)

    if find_keywords(ASSUMPTION_KEYWORDS):
        snippets = _extract_assumptions(text)
        content.assumptions.extend(snippets)

    growths = GROWTH_PATTERN.findall(text)
    if growths:
        content.growth_rates.extend(sorted(set(f"{g}%" for g in growths)))

    return content


def _extract_sample_counts(text: str, max_entries: int = 5) -> List[str]:
    entries: List[str] = []
    for line in text.splitlines():
        lower_line = line.lower()
        if any(dir_kw in lower_line for dir_kw in DIRECTION_KEYWORDS) and any(
            turn_kw in lower_line for turn_kw in TURNING_KEYWORDS
        ):
            numbers = COUNT_PATTERN.findall(line)
            if numbers:
                entries.append(line.strip())
        if len(entries) >= max_entries:
            break
    return entries


def _extract_forecast_volumes(text: str, years: Iterable[str], max_entries: int = 5) -> List[str]:
    entries: List[str] = []
    pattern = re.compile(rf"({'|'.join(map(re.escape, years))}).{{0,60}}?(\d{{1,5}})", re.IGNORECASE)
    for match in pattern.finditer(text):
        snippet = match.group(0).strip()
        entries.append(snippet)
        if len(entries) >= max_entries:
            break
    return entries


def _extract_assumptions(text: str, max_entries: int = 3) -> List[str]:
    snippets: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(keyword in lower_sentence for keyword in ASSUMPTION_KEYWORDS):
            snippets.append(sentence.strip())
        if len(snippets) >= max_entries:
            break
    return snippets


# -----------------------------
# Processing logic
# -----------------------------
def process_file(path: Path) -> Optional[FileSummary]:
    file_name = path.name
    if len(file_name) < 5:
        logging.warning("Skipping %s: filename shorter than 5 characters", file_name)
        return None

    application_id = file_name[:5]
    suffix = path.suffix.lower()
    text = ""
    pages_or_sheets: List[str] = []

    if suffix == ".pdf":
        text, pages_or_sheets = extract_text_pdf(path)
    elif suffix == ".docx":
        text, pages_or_sheets = extract_text_docx(path)
    elif suffix == ".doc":
        text, pages_or_sheets = extract_text_doc(path)
    elif suffix in {".xls", ".xlsx"}:
        text, pages_or_sheets = extract_text_excel(path)
    elif suffix == ".csv":
        text, pages_or_sheets = extract_text_csv(path)
    else:
        logging.debug("Unsupported file type for %s", path)
        return None

    content = detect_traffic_content(text)
    # Store location info
    if suffix in {".xls", ".xlsx", ".csv"}:
        content.sheets.extend(pages_or_sheets)
    else:
        content.pages.extend(pages_or_sheets)

    return FileSummary(application_id=application_id, file_name=file_name, content=content)


def process_folder(folder: Path) -> Dict[str, List[FileSummary]]:
    summaries: Dict[str, List[FileSummary]] = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv")):
                continue
            path = Path(root) / file
            summary = process_file(path)
            if summary is None:
                continue
            summaries.setdefault(summary.application_id, []).append(summary)
    return summaries


# -----------------------------
# Output helpers
# -----------------------------
def format_markdown(summaries: Dict[str, List[FileSummary]]) -> str:
    lines: List[str] = []
    for app_id, files in sorted(summaries.items()):
        lines.append(f"## Application ID: {app_id}\n")
        for summary in files:
            content = summary.content
            lines.append(f"### File: {summary.file_name}\n")
            sheets = ", ".join(content.sheets) if content.sheets else "N/A"
            pages = ", ".join(content.pages) if content.pages else "N/A"
            lines.append(f"- **Sheets with traffic data:** {sheets}")
            lines.append(f"- **Pages with traffic data:** {pages}\n")

            if content.has_data():
                lines.append("**Traffic Counts (Base Conditions)**  ")
                base_years = ", ".join(content.base_years) if content.base_years else "Not specified"
                lines.append(f"- Base year(s): {base_years}  ")
                if content.sample_counts:
                    lines.append("- Sample entries:")
                    for entry in content.sample_counts:
                        lines.append(f"  - {entry}  ")
                else:
                    lines.append("- No specific turning movement entries captured.  ")

                lines.append("\n**Traffic Forecasts**  ")
                forecast_years = ", ".join(content.forecast_years) if content.forecast_years else "Not specified"
                lines.append(f"- Forecast years: {forecast_years}  ")
                if content.forecast_volumes:
                    lines.append("- Sample forecast volumes:")
                    for entry in content.forecast_volumes:
                        lines.append(f"  - {entry}  ")
                else:
                    lines.append("- No forecast volumes captured.  ")
                growth = ", ".join(content.growth_rates) if content.growth_rates else "Not specified"
                lines.append(f"- Growth rate: {growth}  ")

                lines.append("\n**Forecast Assumptions**  ")
                if content.assumptions:
                    for snippet in content.assumptions:
                        lines.append(f"- \"{snippet}\"  ")
                else:
                    lines.append("- No explicit assumptions captured.  ")
            else:
                lines.append("- No traffic volume or forecast-related content detected.  ")

            lines.append("\n---\n")
    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize traffic counts and forecasts from documents.")
    parser.add_argument("input_folder", type=Path, help="Folder containing source documents")
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=Path("traffic_summaries.md"),
        help="Output Markdown file (default: traffic_summaries.md)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input_folder.exists() or not args.input_folder.is_dir():
        logging.error("Input folder %s does not exist or is not a directory", args.input_folder)
        return 1

    summaries = process_folder(args.input_folder)
    if not summaries:
        logging.warning("No files processed or no traffic content detected in the provided folder.")

    markdown = format_markdown(summaries)
    args.output_file.write_text(markdown or "# Traffic Summaries\n", encoding="utf-8")
    logging.info("Summaries written to %s", args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
