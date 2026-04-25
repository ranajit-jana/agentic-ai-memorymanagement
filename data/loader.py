import io
import tempfile
import os
import pandas as pd
import pdfplumber


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [
        "Ticket ID", "Ticket Description", "Ticket Type", "Ticket Subject",
        "Ticket Status", "Resolution", "Ticket Priority", "Ticket Channel",
        "Customer Satisfaction Rating", "Date of Purchase",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df["text"] = (
        df["Ticket Subject"].fillna("") + " " + df["Ticket Description"].fillna("")
    ).str.strip()
    return df


def load_pdf(path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text.strip()})
    return pages


def load_txt(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    segments = [s.strip() for s in content.split("\n\n") if s.strip()]
    return segments


def handle_uploaded_file(uploaded_file) -> dict:
    """
    Accept a file path (str) or a Streamlit UploadedFile object.
    Routes to the correct loader based on file extension.
    Returns {"type": str, "data": DataFrame | list, "filename": str}
    """
    if isinstance(uploaded_file, str):
        filename = os.path.basename(uploaded_file)
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            return {"type": "csv", "data": load_csv(uploaded_file), "filename": filename}
        elif ext == "pdf":
            return {"type": "pdf", "data": load_pdf(uploaded_file), "filename": filename}
        elif ext == "txt":
            return {"type": "txt", "data": load_txt(uploaded_file), "filename": filename}
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

    # Streamlit UploadedFile object
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower()
    raw_bytes = uploaded_file.read()

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(raw_bytes))
        cols = [
            "Ticket ID", "Ticket Description", "Ticket Type", "Ticket Subject",
            "Ticket Status", "Resolution", "Ticket Priority", "Ticket Channel",
            "Customer Satisfaction Rating", "Date of Purchase",
        ]
        df = df[[c for c in cols if c in df.columns]]
        df["text"] = (
            df["Ticket Subject"].fillna("") + " " + df["Ticket Description"].fillna("")
        ).str.strip()
        return {"type": "csv", "data": df, "filename": filename}

    elif ext == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            pages = load_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
        return {"type": "pdf", "data": pages, "filename": filename}

    elif ext == "txt":
        content = raw_bytes.decode("utf-8")
        segments = [s.strip() for s in content.split("\n\n") if s.strip()]
        return {"type": "txt", "data": segments, "filename": filename}

    else:
        raise ValueError(f"Unsupported file type: .{ext}")
