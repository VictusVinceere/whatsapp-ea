"""
Index a folder of documents so the assistant can answer from them.

    uv run python -m app.ingest ~/notes
    uv run python -m app.ingest ~/notes --ext .md .txt

Re-running is safe: each file's chunks are keyed by its path, and
ingest_document deletes a source's old chunks before writing new ones, so
an edited file is replaced rather than duplicated.

Text files only for now. PDF and docx need an extractor -- pypdf and
python-docx are the usual choices, and both are a bigger job than they
look because layout, tables and columns all affect how the text chunks.
"""

import argparse
import asyncio
import pathlib

from app.db import DEFAULT_TENANT_ID
from app.rag import ingest_document

DEFAULT_EXTENSIONS = (".md", ".txt", ".markdown", ".rst")


async def ingest_folder(folder: pathlib.Path, extensions: tuple[str, ...]) -> None:
    files = sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )
    if not files:
        print(f"no {'/'.join(extensions)} files under {folder}")
        return

    total = 0
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            # One unreadable file shouldn't abandon the whole folder.
            print(f"  skipped {path.name}: {type(exc).__name__}")
            continue

        chunks = await ingest_document(
            DEFAULT_TENANT_ID,
            # Full path as the key: two files can share a name in
            # different folders, and re-ingesting must replace the right one.
            source_id=str(path.resolve()),
            source_name=path.stem,
            content=content,
        )
        total += chunks
        print(f"  {path.name:<40} {chunks:>3} chunks")

    print(f"\nindexed {len(files)} files, {total} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents for RAG.")
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTENSIONS))
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"not a directory: {args.folder}")

    asyncio.run(
        ingest_folder(args.folder, tuple(e.lower() for e in args.ext))
    )


if __name__ == "__main__":
    main()
