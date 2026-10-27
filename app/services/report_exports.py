from __future__ import annotations

import csv
import io
import re
from typing import Any, Iterable

from app.schemas.charts import ChartSpec
from app.schemas.reports import ReportRead


_CSV_FORMULA_PREFIX_RE = re.compile(r"^[=+\-@]")


def _safe_filename(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return cleaned or fallback


def _csv_safe_cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    # Prevent CSV formula injection in Excel-like clients.
    # Excel treats leading whitespace before '=' as a formula trigger.
    candidate = (text or "").lstrip()
    if _CSV_FORMULA_PREFIX_RE.match(candidate):
        text = "'" + (text or "")
    return text


def _write_csv(rows: Iterable[Iterable[Any]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        writer.writerow([_csv_safe_cell(cell) for cell in row])
    return buf.getvalue()


def _chart_title(spec: ChartSpec) -> str:
    title = getattr(spec, "title", None)
    return str(title) if title else ""


def export_report_results_csv(report: ReportRead) -> str:
    key_findings = "; ".join(report.key_findings or [])
    rows: list[list[Any]] = [
        [
            "report_id",
            "data_source_id",
            "question",
            "summary",
            "key_findings",
            "created_at",
            "updated_at",
        ],
        [
            report.id,
            report.data_source_id,
            report.question,
            report.summary,
            key_findings,
            report.created_at.isoformat(),
            report.updated_at.isoformat(),
        ],
    ]
    return _write_csv(rows)


def export_report_charts_csv(report: ReportRead) -> str:
    chart_specs = report.chart_specifications or []
    rows: list[list[Any]] = []

    if not chart_specs:
        rows.append(["chart_type"])
        return _write_csv(rows)

    # Flatten each chart spec into rows by its native shape.
    for idx, spec in enumerate(chart_specs, start=1):
        spec_type = getattr(spec, "type", None)
        title = _chart_title(spec)

        if spec_type in {"line", "bar", "scatter"}:
            rows.append(["chart_index", "chart_type", "title", "x_axis", "y_axis", "x", "y"])
            x_axis = getattr(spec, "x_axis", None)
            y_axis = getattr(spec, "y_axis", None)
            xs = getattr(spec, "x", None) or []
            ys = getattr(spec, "y", None) or []
            for i in range(min(len(xs), len(ys))):
                rows.append([idx, spec_type, title, x_axis, y_axis, xs[i], ys[i]])
            continue

        if spec_type == "pie":
            rows.append(["chart_index", "chart_type", "title", "label", "value"])
            labels = getattr(spec, "labels", None) or []
            values = getattr(spec, "values", None) or []
            for i in range(min(len(labels), len(values))):
                rows.append([idx, spec_type, title, labels[i], values[i]])
            continue

        if spec_type == "table":
            columns = list(getattr(spec, "columns", None) or [])
            rows.append(["chart_index", "chart_type", "title", *columns])
            table_rows = getattr(spec, "rows", None) or []
            for r in table_rows[:1000]:
                padded = list(r)[: len(columns)]
                padded.extend([""] * (len(columns) - len(padded)))
                rows.append([idx, spec_type, title, *padded])
            continue

        # Unknown chart shapes: export nothing but keep a breadcrumb.
        rows.append(["chart_index", "chart_type", "title"])
        rows.append([idx, spec_type or "", title])

    return _write_csv(rows)


def _pdf_escape_text(text: str) -> str:
    # Minimal escaping for PDF string literal content.
    text = (text or "")
    # Content stream strings should not contain raw line breaks/control chars.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    # Keep it ASCII-ish to avoid font encoding issues.
    text = "".join(ch if ord(ch) < 128 else "?" for ch in text)
    return text


def _pdf_obj(obj_id: int, body: str) -> bytes:
    return f"{obj_id} 0 obj\n{body}\nendobj\n".encode("latin-1")


def build_report_pdf_bytes(report: ReportRead) -> bytes:
    # Basic, text-only PDF. Chart data is included as a compact textual summary.
    lines: list[str] = []
    lines.append(f"Report: {report.question}".strip())
    lines.append(f"Summary: {report.summary}".strip())
    lines.append("")
    lines.append("Key Findings:")
    if report.key_findings:
        for item in report.key_findings[:50]:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")
    lines.append("")

    chart_specs = report.chart_specifications or []
    if not chart_specs:
        lines.append("Charts: (none)")
    else:
        lines.append("Charts:")
        for i, spec in enumerate(chart_specs, start=1):
            spec_type = getattr(spec, "type", "")
            title = _chart_title(spec)
            x_axis = getattr(spec, "x_axis", None)
            y_axis = getattr(spec, "y_axis", None)
            label = f"{i}) {spec_type}"
            if title:
                label += f" - {title}"
            if x_axis or y_axis:
                label += f" ({x_axis or ''}/{y_axis or ''})"
            lines.append(label)

            if spec_type in {"line", "bar", "scatter"}:
                xs = getattr(spec, "x", None) or []
                ys = getattr(spec, "y", None) or []
                points = min(len(xs), len(ys), 20)
                for j in range(points):
                    lines.append(f"  - {xs[j]}: {ys[j]}")
            elif spec_type == "pie":
                labels = getattr(spec, "labels", None) or []
                values = getattr(spec, "values", None) or []
                items = min(len(labels), len(values), 20)
                for j in range(items):
                    lines.append(f"  - {labels[j]}: {values[j]}")
            elif spec_type == "table":
                cols = list(getattr(spec, "columns", None) or [])
                table_rows = getattr(spec, "rows", None) or []
                lines.append(f"  Columns: {', '.join(str(c) for c in cols)}")
                for r in table_rows[:5]:
                    lines.append(f"  Row: {', '.join(str(x) for x in r[: len(cols)])}")

            lines.append("")

    # Pagination: one page only; truncate if content is too long.
    page_w, page_h = 612, 792  # 8.5x11 inches at 72dpi
    margin_left = 54
    margin_top = 60
    font_size = 12
    line_height = 14
    max_lines = int((page_h - margin_top - 60) / line_height)
    lines = lines[:max_lines]

    content_cmds: list[str] = []
    # Use text matrix (Tm) per line.
    for i, line in enumerate(lines):
        y = page_h - margin_top - i * line_height
        content_cmds.append(
            f"BT /F1 {font_size} Tf 1 0 0 1 {margin_left} {y} Tm ({_pdf_escape_text(line)}) Tj ET"
        )
    content = "\n".join(content_cmds)
    content_bytes = content.encode("latin-1", errors="replace")

    # Build PDF objects with computed xref offsets.
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    objects: list[bytes] = []
    objects.append(_pdf_obj(1, "<< /Type /Catalog /Pages 2 0 R >>"))
    objects.append(_pdf_obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    objects.append(
        _pdf_obj(
            3,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w} {page_h}] "
            f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        )
    )
    objects.append(_pdf_obj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objects.append(
        b"5 0 obj\n"
        + f"<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1")
        + content_bytes
        + b"\nendstream\nendobj\n"
    )

    # Xref table.
    offset = len(header)
    xref_entries = [b"0000000000 65535 f \n"]
    full_pdf = bytearray(header)
    for obj in objects:
        offset_str = str(offset).zfill(10)
        xref_entries.append(offset_str.encode("ascii") + b" 00000 n \n")
        full_pdf.extend(obj)
        offset += len(obj)

    xref_start = len(full_pdf)
    xref = (
        b"xref\n"
        + f"0 {len(xref_entries)}\n".encode("ascii")
        + b"".join(xref_entries)
        + b"trailer\n<< /Size "
        + str(len(xref_entries)).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF\n"
    )
    full_pdf.extend(xref)
    pdf = bytes(full_pdf)
    # Fail fast on obvious PDF structure corruption.
    if (
        b"xref" not in pdf
        or b"trailer" not in pdf
        or b"startxref" not in pdf
        or not pdf.rstrip().endswith(b"%%EOF")
    ):
        raise ValueError("PDF generation failed sanity checks")
    return pdf

