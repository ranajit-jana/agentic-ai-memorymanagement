import re
import pandas as pd
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _sliding_window(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def _split_multi_turn(text: str) -> list[str] | None:
    """Split conversation into per-turn chunks when turn markers are present."""
    pattern = re.compile(r'((?:Customer|Agent|User|Support|Rep)\s*:)', re.IGNORECASE)
    parts = pattern.split(text)
    if len(parts) <= 2:
        return None
    turns = []
    for i in range(1, len(parts) - 1, 2):
        turn = (parts[i] + parts[i + 1]).strip()
        if turn:
            turns.append(turn)
    return turns if len(turns) > 1 else None


def chunk_csv_tickets(df: pd.DataFrame) -> list[dict]:
    """
    Each ticket row → one or more chunks.
    Multi-turn conversations are split per turn.
    Long single-block descriptions use a sliding window.
    """
    chunks = []
    for _, row in df.iterrows():
        ticket_id = str(row.get("Ticket ID", ""))
        text = str(row.get("text", "")).strip()
        metadata = {
            "source":        "csv_ticket",
            "ticket_id":     ticket_id,
            "ticket_type":   str(row.get("Ticket Type", "")),
            "priority":      str(row.get("Ticket Priority", "")),
            "status":        str(row.get("Ticket Status", "")),
            "channel":       str(row.get("Ticket Channel", "")),
            "date_of_purchase": str(row.get("Date of Purchase", "")),
            "satisfaction":  str(row.get("Customer Satisfaction Rating", "")),
        }

        turns = _split_multi_turn(text)
        if turns:
            for i, turn in enumerate(turns):
                chunks.append({
                    "id":     f"ticket_{ticket_id}_turn_{i}",
                    "text":   turn,
                    "source": "csv_ticket",
                    "metadata": {**metadata, "turn_index": str(i)},
                })
        elif len(text) > CHUNK_SIZE:
            for i, seg in enumerate(_sliding_window(text)):
                chunks.append({
                    "id":     f"ticket_{ticket_id}_chunk_{i}",
                    "text":   seg,
                    "source": "csv_ticket",
                    "metadata": {**metadata, "chunk_index": str(i)},
                })
        else:
            chunks.append({
                "id":     f"ticket_{ticket_id}",
                "text":   text,
                "source": "csv_ticket",
                "metadata": metadata,
            })
    return chunks


def chunk_pdf_pages(pages: list[dict], source_name: str = "pdf_policy") -> list[dict]:
    """
    Chunk PDF by section/topic.
    Each page is treated as one policy section.
    If the page text exceeds CHUNK_SIZE it is further split with a sliding window.
    The section title is extracted from the first non-empty line.
    """
    chunks = []
    for page_dict in pages:
        page_num = page_dict["page"]
        text = page_dict["text"].strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        section_title = lines[0] if lines else f"Page {page_num}"

        metadata = {
            "source":     source_name,
            "page":       str(page_num),
            "section":    section_title,
            "chunk_text": text[:1000],  # stored for retrieval in find_policy_reference
        }

        if len(text) <= CHUNK_SIZE:
            chunks.append({
                "id":     f"{source_name}_page_{page_num}",
                "text":   text,
                "source": source_name,
                "metadata": metadata,
            })
        else:
            for i, seg in enumerate(_sliding_window(text)):
                chunks.append({
                    "id":     f"{source_name}_page_{page_num}_chunk_{i}",
                    "text":   seg,
                    "source": source_name,
                    "metadata": {**metadata, "chunk_index": str(i), "chunk_text": seg[:1000]},
                })
    return chunks


def chunk_txt_segments(segments: list[str], source_name: str = "txt_notes") -> list[dict]:
    """
    Chunk internal notes / chat logs.
    Each segment (double-newline-separated entry) is split per agent turn if markers exist,
    otherwise chunked by size.
    Agent/Date/Ticket metadata is parsed from the header line.
    """
    chunks = []
    for i, segment in enumerate(segments):
        first_line = segment.split("\n")[0]
        metadata = {"source": source_name, "segment_index": str(i), "chunk_text": segment[:1000]}

        for key, pat in [
            ("agent",      r'Agent:\s*(\w+)'),
            ("date",       r'Date:\s*([\d-]+)'),
            ("ticket_ref", r'Ticket:\s*(#?\w+)'),
        ]:
            m = re.search(pat, first_line)
            if m:
                metadata[key] = m.group(1)

        turns = _split_multi_turn(segment)
        if turns:
            for j, turn in enumerate(turns):
                chunks.append({
                    "id":     f"{source_name}_{i}_turn_{j}",
                    "text":   turn,
                    "source": source_name,
                    "metadata": {**metadata, "turn_index": str(j)},
                })
        elif len(segment) > CHUNK_SIZE:
            for j, seg in enumerate(_sliding_window(segment)):
                chunks.append({
                    "id":     f"{source_name}_{i}_chunk_{j}",
                    "text":   seg,
                    "source": source_name,
                    "metadata": {**metadata, "chunk_index": str(j)},
                })
        else:
            chunks.append({
                "id":     f"{source_name}_{i}",
                "text":   segment,
                "source": source_name,
                "metadata": metadata,
            })
    return chunks
