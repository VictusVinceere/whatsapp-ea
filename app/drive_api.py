"""
Google Drive: list, download, upload.

Drive has two kinds of file and they are fetched differently. A PDF or
docx you uploaded is a *binary* file and comes back with alt=media. A
Google Doc, Sheet or Slide has no bytes of its own -- it is a database
row Google renders on demand -- so it must be *exported* to a chosen
format instead. Calling alt=media on one returns
"fileNotExportable"; the two paths are not interchangeable.
"""

import httpx
import structlog

log = structlog.get_logger()

FILES = "https://www.googleapis.com/drive/v3/files"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"

# Google-native types and what to export them as. Plain text keeps the
# extraction path simple -- we only want words to embed, not layout.
EXPORT_AS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# Binary types we can extract text from. Anything else (images, video,
# zip) is listed but skipped -- embedding a filename alone is noise.
READABLE = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


async def list_files(access_token: str, max_results: int = 25) -> list[dict]:
    """Files the user can read, newest first, excluding trash and folders."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            FILES,
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "pageSize": max_results,
                "orderBy": "modifiedTime desc",
                "q": "trashed = false and mimeType != 'application/vnd.google-apps.folder'",
                "fields": "files(id,name,mimeType,modifiedTime,size)",
            },
        )
        response.raise_for_status()
        return response.json().get("files", [])


def is_readable(mime_type: str) -> bool:
    return mime_type in EXPORT_AS or mime_type in READABLE


def suggested_name(name: str, mime_type: str) -> str:
    """A filename the extractor can dispatch on.

    A Google Doc is called "Q3 Report" with no extension, and exports as
    plain text -- so it needs a .txt suffix or extract_text would try to
    parse it as something else.
    """
    if mime_type in EXPORT_AS:
        return f"{name}.txt" if EXPORT_AS[mime_type] != "text/csv" else f"{name}.csv"
    suffix = READABLE.get(mime_type, "")
    return name if name.lower().endswith(suffix) else f"{name}{suffix}"


async def fetch_file(access_token: str, file_id: str, mime_type: str) -> bytes:
    """Raw bytes, via export for Google-native files and alt=media otherwise."""
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        if mime_type in EXPORT_AS:
            response = await client.get(
                f"{FILES}/{file_id}/export",
                headers=headers,
                params={"mimeType": EXPORT_AS[mime_type]},
            )
        else:
            response = await client.get(
                f"{FILES}/{file_id}", headers=headers, params={"alt": "media"}
            )
        response.raise_for_status()
        return response.content


async def upload_file(
    access_token: str, name: str, data: bytes, mime_type: str
) -> dict:
    """Upload bytes to the user's Drive.

    Multipart: one request carrying the metadata (the name) and the bytes
    together. The simple upload endpoint takes bytes but no name, which
    leaves files called "Untitled".
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            UPLOAD,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
            files={
                "metadata": (
                    None,
                    f'{{"name": "{name}"}}',
                    "application/json; charset=UTF-8",
                ),
                "file": (name, data, mime_type),
            },
        )
        response.raise_for_status()
        return response.json()


def describe_files(files: list[dict]) -> str:
    if not files:
        return "(no files)"
    lines = []
    for f in files:
        mark = "" if is_readable(f.get("mimeType", "")) else "  [not text]"
        lines.append(f"- {f.get('name')}  ({f.get('modifiedTime','?')[:10]}){mark}")
    return "\n".join(lines)
