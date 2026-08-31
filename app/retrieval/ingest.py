"""
Read knowledge-base Markdown files and prepare them for search.
Reads metadata at the top, splits documents into sections, keeps track of everything.
"""

import os
import re
from typing import Any
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import KNOWLEDGE_BASE_DIR


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Read the metadata block at the top of a file and return (metadata, body)."""
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    front = text[3:end].strip()
    body = text[end + 3:].strip()

    metadata: dict[str, Any] = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            metadata[key.strip()] = val.strip()

    return metadata, body


def chunk_by_headings(body: str, metadata: dict[str, Any], filename: str) -> list[dict[str, Any]]:
    """
    Split a file into sections at each heading (##).
    Each section keeps the file's metadata plus its own heading.
    """
    # Split on ## headings
    sections = re.split(r'\n(?=## )', body)

    chunks: list[dict[str, Any]] = []

    # First section may be intro text before the first ##
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # Detect heading
        if lines[0].startswith("## "):
            heading = lines[0][3:].strip()
            content = "\n".join(lines[1:]).strip()
        elif lines[0].startswith("# "):
            heading = lines[0][2:].strip()
            content = "\n".join(lines[1:]).strip()
        else:
            heading = metadata.get("title", filename)
            content = "\n".join(lines).strip()

        if not content:
            continue

        chunks.append({
            "filename": filename,
            "document_id": metadata.get("document_id", ""),
            "title": metadata.get("title", ""),
            "heading": heading,
            "status": metadata.get("status", "active"),
            "audience": metadata.get("audience", "customer"),
            "policy_authority": metadata.get("policy_authority", "none"),
            "effective_date": metadata.get("effective_date", ""),
            "supersedes": metadata.get("supersedes", ""),
            "superseded_by": metadata.get("superseded_by", ""),
            "customer_answering": metadata.get("customer_answering", "true").lower() != "false",
            "text": content,
            "source_ref": f"{filename} — {heading}",
        })

    return chunks


def load_all_chunks() -> list[dict[str, Any]]:
    """Load every file from the knowledge-base directory and return all chunks."""
    all_chunks: list[dict[str, Any]] = []

    kb_dir = os.path.abspath(KNOWLEDGE_BASE_DIR)
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"Knowledge-base directory not found: {kb_dir}")

    for fname in sorted(os.listdir(kb_dir)):
        if not fname.endswith(".md"):
            continue

        path = os.path.join(kb_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        metadata, body = parse_front_matter(raw)
        chunks = chunk_by_headings(body, metadata, fname)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
    for c in chunks:
        print(f"[{c['status']}/{c['audience']}/{c['policy_authority']}] "
              f"{c['filename']} — {c['heading']} ({len(c['text'])} chars)")
    print(f"\nTotal chunks: {len(chunks)}")
