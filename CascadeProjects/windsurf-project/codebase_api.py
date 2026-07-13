#!/usr/bin/env python3
"""Read-only codebase context API for Weaver.

The service is intentionally narrow: it exposes capped snippets from source and
documentation files so Weaver can reason about her own deployed codebase without
getting shell access, write access, model files, vault memory, or secrets.
"""

from __future__ import annotations

import os
import re
import time
import html
import ipaddress
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from fastapi import Body, FastAPI, HTTPException, Query


def _default_project_root() -> Path:
    override = os.environ.get("WEAVER_CODEBASE_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    workspace = here.parents[2] if len(here.parents) > 2 else here.parent
    if (workspace / "CascadeProjects").is_dir():
        return workspace
    return here.parent


PROJECT_ROOT = _default_project_root()
MAX_FILE_BYTES = int(os.environ.get("WEAVER_CODEBASE_MAX_FILE_BYTES", str(256 * 1024)))
DEFAULT_CONTEXT_CHARS = int(os.environ.get("WEAVER_CODEBASE_CONTEXT_CHARS", "6000"))
MAX_CONTENT_SEARCH_FILES = int(os.environ.get("WEAVER_CODEBASE_SEARCH_FILES", "180"))
MATCH_READ_CHARS = min(
    int(os.environ.get("WEAVER_CODEBASE_MATCH_CHARS", str(MAX_FILE_BYTES))),
    MAX_FILE_BYTES,
)
INTERNET_MAX_BYTES = int(os.environ.get("WEAVER_INTERNET_MAX_BYTES", str(768 * 1024)))
INTERNET_TIMEOUT_SECONDS = float(os.environ.get("WEAVER_INTERNET_TIMEOUT_SECONDS", "5.0"))

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "Nexus_Vault",
    "Weaver_Vault",
    "SYPHER_VAULT",
    "models",
    "3b",
    "downloads",
    "renders",
    "assets",
    "images",
    "logs",
    "torchinductor_ydn",
    "weaver_fracture_1B_lora",
    "weaver_merged_1B",
    "Quantum Computing Architecture_files",
}
EXCLUDED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "ghost_key.json",
    "token.json",
    "terraform.tfvars",
    "tfplan",
    "id_rsa",
    "id_ed25519",
}
EXCLUDED_NAME_PARTS = {
    "credential",
    "client_secret",
    "private_key",
    "secret_key",
    "access_token",
    "refresh_token",
}
ALLOWED_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".mjs",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".yml",
    ".yaml",
    ".toml",
    ".tf",
    ".service",
    ".example",
}
ALLOWED_NAMES = {"Dockerfile", "Makefile", "Caddyfile", "README", "CLAUDE.md"}
SECRET_PATTERNS = [
    re.compile(r"(?i)\b(aws_access_key_id|aws_secret_access_key)\s*=\s*[^\s]+"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passphrase|credential)([\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
]

app = FastAPI(title="Weaver Codebase API", version="1.0.0")
_CODE_FILE_CACHE: list[Path] = []
_CODE_FILE_CACHE_AT = 0.0
_CODE_FILE_CACHE_SECONDS = float(os.environ.get("WEAVER_CODEBASE_INDEX_TTL", "15"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 - stdlib API
        return None


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") and part not in {".gitignore"} for part in path.parts)


def _is_allowed_file(path: Path) -> bool:
    name = path.name
    lower_name = name.lower()
    if name in EXCLUDED_NAMES:
        return False
    if any(part in lower_name for part in EXCLUDED_NAME_PARTS):
        return False
    if _is_hidden(path.relative_to(PROJECT_ROOT)):
        return False
    if any(part in EXCLUDED_DIRS for part in path.relative_to(PROJECT_ROOT).parts[:-1]):
        return False
    if name in ALLOWED_NAMES or name.startswith("Dockerfile"):
        return True
    return path.suffix.lower() in ALLOWED_SUFFIXES


def _load_git_tracked_paths() -> tuple[str, ...] | None:
    try:
        tracked = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files", "-z"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if tracked.returncode != 0 or not tracked.stdout:
        return None
    return tuple(os.fsdecode(raw) for raw in tracked.stdout.split(b"\0") if raw)


# Loaded once on the importing thread. Request workers never fork a Git process.
_GIT_TRACKED_PATHS = _load_git_tracked_paths()


def _resolve_repo_path(rel_path: str) -> Path:
    if not rel_path:
        raise HTTPException(status_code=400, detail="path is required")
    target = (PROJECT_ROOT / rel_path).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path escapes codebase root") from exc
    if not target.is_file() or not _is_allowed_file(target):
        raise HTTPException(status_code=404, detail="file not exposed")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="file too large for codebase API")
    return target


def _iter_code_files(limit: int = 1200) -> Iterable[Path]:
    global _CODE_FILE_CACHE, _CODE_FILE_CACHE_AT
    now = time.monotonic()
    if not _CODE_FILE_CACHE or now - _CODE_FILE_CACHE_AT >= _CODE_FILE_CACHE_SECONDS:
        discovered: list[Path] = []
        if _GIT_TRACKED_PATHS is not None:
            for rel_path in _GIT_TRACKED_PATHS:
                path = PROJECT_ROOT / rel_path
                try:
                    path.resolve().relative_to(PROJECT_ROOT)
                    if not path.is_file():
                        continue
                    if not _is_allowed_file(path):
                        continue
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                    discovered.append(path)
                    if len(discovered) >= 1200:
                        break
                except (OSError, ValueError):
                    continue
        else:
            for root, dirs, files in os.walk(PROJECT_ROOT):
                root_path = Path(root)
                dirs[:] = [
                    d for d in sorted(dirs)
                    if d not in EXCLUDED_DIRS and not d.startswith(".")
                ]
                for name in sorted(files):
                    path = root_path / name
                    try:
                        if not _is_allowed_file(path):
                            continue
                        if path.stat().st_size > MAX_FILE_BYTES:
                            continue
                        discovered.append(path)
                        if len(discovered) >= 1200:
                            break
                    except OSError:
                        continue
                if len(discovered) >= 1200:
                    break
        _CODE_FILE_CACHE = discovered
        _CODE_FILE_CACHE_AT = now
    yield from _CODE_FILE_CACHE[:limit]


def _redact(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 3:
            redacted = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _compact_web_text(text: str, max_chars: int) -> str:
    return re.sub(r"\s+", " ", _redact(text)).strip()[:max_chars]


def _public_host(hostname: str) -> str:
    host = (hostname or "").strip().strip("[]").rstrip(".")
    if not host:
        raise HTTPException(status_code=400, detail="url host is required")
    lower = host.lower()
    if lower in {"localhost", "metadata.google.internal"} or lower.endswith((".local", ".internal")):
        raise HTTPException(status_code=403, detail="host is not public")

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="host does not resolve") from exc

    for info in infos:
        raw_ip = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="unparseable host address") from exc
        if not ip.is_global:
            raise HTTPException(status_code=403, detail="private, local, or reserved hosts are blocked")
    return host


def _public_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="url is required")
    if "://" not in value:
        value = "https://" + value
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="only http/https urls are allowed")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=403, detail="credentialed urls are blocked")
    _public_host(parsed.hostname or "")
    return urllib.parse.urlunparse(parsed._replace(fragment=""))


def _decode_web_body(raw: bytes, content_type: str) -> str:
    match = re.search(r"charset=([A-Za-z0-9_.:-]+)", content_type or "", re.I)
    charset = match.group(1) if match else "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _html_to_text(text: str, max_chars: int) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    title = _compact_web_text(html.unescape(title_match.group(1)), 160) if title_match else ""
    stripped = re.sub(r"(?is)<(script|style|noscript|svg|canvas|template)\b.*?</\1>", " ", text)
    stripped = re.sub(r"(?is)<!--.*?-->", " ", stripped)
    stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
    return title, _compact_web_text(html.unescape(stripped), max_chars)


def _fetch_public_raw(raw_url: str, redirects: int = 3) -> tuple[str, int, str, bytes]:
    url = _public_url(raw_url)
    opener = urllib.request.build_opener(_NoRedirect)
    last_url = url
    for _ in range(max(0, redirects) + 1):
        req = urllib.request.Request(
            last_url,
            headers={
                "User-Agent": "WeaverReach/1.0 (+https://weaverv3.com; read-only)",
                "Accept": "text/html,text/plain,application/json,application/xml,application/rss+xml;q=0.8,*/*;q=0.1",
            },
            method="GET",
        )
        try:
            with opener.open(req, timeout=INTERNET_TIMEOUT_SECONDS) as resp:
                status = resp.status
                content_type = (resp.headers.get("content-type") or "").lower()
                raw = resp.read(INTERNET_MAX_BYTES + 1)
                break
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                last_url = _public_url(urllib.parse.urljoin(last_url, exc.headers["Location"]))
                continue
            if exc.code >= 400:
                raise HTTPException(status_code=502, detail=f"upstream returned {exc.code}") from exc
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise HTTPException(status_code=502, detail="public web fetch failed") from exc
    else:
        raise HTTPException(status_code=508, detail="too many redirects")

    if len(raw) > INTERNET_MAX_BYTES:
        raise HTTPException(status_code=413, detail="response too large")
    allowed = (
        content_type.startswith("text/")
        or "json" in content_type
        or "xml" in content_type
        or "rss" in content_type
        or "atom" in content_type
    )
    if content_type and not allowed:
        raise HTTPException(status_code=415, detail="non-text content is blocked")

    return last_url, status, content_type or "unknown", raw


def fetch_public_page(raw_url: str, max_chars: int = 6000, redirects: int = 3) -> dict:
    last_url, status, content_type, raw = _fetch_public_raw(raw_url, redirects=redirects)
    body = _decode_web_body(raw, content_type)
    if "html" in content_type or "<html" in body[:600].lower():
        title, text = _html_to_text(body, max_chars)
    else:
        title, text = "", _compact_web_text(body, max_chars)
    return {
        "url": last_url,
        "status": status,
        "content_type": content_type or "unknown",
        "title": title,
        "text": text,
        "policy": "read-only public GET; private/local/reserved hosts, credentials, binary content, large responses, and redirects to blocked hosts are denied",
    }


def search_public_web(query: str, max_results: int = 5) -> list[dict]:
    query = _compact_web_text(query, 240)
    if not query:
        return []
    search_url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    _, _, content_type, raw_body = _fetch_public_raw(search_url)
    body = _decode_web_body(raw_body, content_type)
    results: list[dict] = []
    seen: set[str] = set()
    for href, title_html in re.findall(r'(?is)<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', body):
        href = html.unescape(href)
        if "duckduckgo.com/l/?" in href or href.startswith("/l/?"):
            params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = params.get("uddg", [href])[0]
        clean = html.unescape(urllib.parse.urljoin(search_url, href)).rstrip(").,]")
        try:
            public = _public_url(clean)
        except HTTPException:
            continue
        if "duckduckgo.com" in urllib.parse.urlparse(public).netloc or public in seen:
            continue
        seen.add(public)
        title, snippet = _html_to_text(title_html, 420)
        results.append({"title": title or snippet[:160], "url": public, "snippet": snippet})
        if len(results) >= max_results:
            break
    if not results:
        _, fallback_text = _html_to_text(body, 900)
        results.append({"title": f"Search page for {query}", "url": search_url, "snippet": fallback_text})
    return results[:max_results]


def _read_text(path: Path, max_chars: int) -> str:
    with path.open("rb") as f:
        raw = f.read(min(MAX_FILE_BYTES, max_chars * 4))
    text = raw.decode("utf-8", errors="replace")
    return _redact(text[:max_chars])


def _context_chunk(path: Path, max_chars: int, matches: list[dict] | None = None) -> str:
    header = f"### {_rel(path)}\n"
    if matches:
        source_lines = _read_text(path, MATCH_READ_CHARS).splitlines()
        selected_lines: list[int] = []
        seen_lines: set[int] = set()
        for match in matches:
            line_index = max(0, int(match["line"]) - 1)
            for selected in range(max(0, line_index - 1), min(len(source_lines), line_index + 2)):
                if selected not in seen_lines:
                    seen_lines.add(selected)
                    selected_lines.append(selected)
        excerpts = "\n".join(
            f"L{line_index + 1}: {source_lines[line_index]}"
            for line_index in selected_lines
        )
        return (header + "Relevant source excerpts:\n" + excerpts)[:max_chars]
    body_chars = max(600, max_chars - len(header))
    return (header + _read_text(path, body_chars))[:max_chars]


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _file_meta(path: Path) -> dict:
    st = path.stat()
    return {
        "path": _rel(path),
        "size": st.st_size,
        "modified": int(st.st_mtime),
    }


def _matches_for_file(path: Path, terms: list[str], max_matches: int = 16) -> list[dict]:
    try:
        text = _read_text(path, MATCH_READ_CHARS)
    except OSError:
        return []
    candidates = []
    for idx, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        matched = [term for term in terms if term in low]
        if matched:
            score = sum(4 if "_" in term or "." in term else 1 for term in matched)
            if re.search(r"\bname\s*=", line):
                score += 12
            if re.match(r'^\s*"[A-Z]{1,2}"\s*:', line):
                score += 12
            candidates.append({
                "line": idx,
                "text": line.strip()[:240],
                "score": score,
                "matched": matched,
            })
    ranked = sorted(candidates, key=lambda item: (-item["score"], item["line"]))
    selected: list[dict] = []
    selected_lines: set[int] = set()
    coverage_terms = [
        term for term in terms
        if "_" in term or "." in term or term in {"cortexrouted"}
    ]
    for term in coverage_terms:
        match = next((item for item in ranked if term in item["matched"]), None)
        if match is not None and match["line"] not in selected_lines:
            selected.append(match)
            selected_lines.add(match["line"])
            if len(selected) >= max_matches:
                break
    for item in ranked:
        if len(selected) >= max_matches:
            break
        if item["line"] not in selected_lines:
            selected.append(item)
            selected_lines.add(item["line"])
    return [
        {
            "line": item["line"],
            "text": item["text"],
            "score": item["score"],
            "matched": item["matched"],
        }
        for item in selected
    ]


def build_manifest(limit: int = 200) -> dict:
    files = [_file_meta(path) for path in _iter_code_files(limit=limit)]
    return {
        "root": str(PROJECT_ROOT),
        "generated_at": int(time.time()),
        "file_count": len(files),
        "files": files,
        "policy": {
            "mode": "read_only",
            "max_file_bytes": MAX_FILE_BYTES,
            "excluded_dirs": sorted(EXCLUDED_DIRS),
        },
    }


def search_codebase(query: str, max_files: int = 10) -> list[dict]:
    raw_terms = re.findall(r"[A-Za-z0-9_./-]{2,}", query)[:40]
    terms: list[str] = []
    for raw_term in raw_terms:
        lowered = raw_term.lower()
        variants = [lowered, *lowered.split("-")]
        if lowered.endswith("ing") and len(lowered) > 6:
            variants.append(lowered[:-3])
        if lowered.endswith("ed") and len(lowered) > 5:
            variants.append(lowered[:-1])
        for variant in variants:
            if len(variant) >= 2 and variant not in terms:
                terms.append(variant)
    if not terms:
        return []

    path_hits = []
    content_candidates = []
    has_exact_path = False
    for path in _iter_code_files():
        rel = _rel(path).lower()
        name = path.name.lower()
        exact_path = any(term == name or rel.endswith(f"/{term}") for term in terms)
        has_exact_path = has_exact_path or exact_path
        path_score = sum(
            100 if term == name or rel.endswith(f"/{term}") else 2
            for term in terms
            if term in rel
        )
        if path_score:
            matches = _matches_for_file(path, terms)
            path_hits.append({
                **_file_meta(path),
                "score": path_score + sum(match.get("score", 1) for match in matches),
                "matches": matches,
            })
        else:
            content_candidates.append(path)

    path_hits.sort(key=lambda item: (-item["score"], item["path"]))
    results = path_hits
    content_scan_limit = min(MAX_CONTENT_SEARCH_FILES, 64) if has_exact_path else MAX_CONTENT_SEARCH_FILES
    for path in content_candidates[:content_scan_limit]:
        matches = _matches_for_file(path, terms)
        if matches:
            results.append({
                **_file_meta(path),
                "score": sum(match.get("score", 1) for match in matches),
                "matches": matches,
            })
    results.sort(key=lambda item: (-item["score"], item["path"]))
    return results[:max_files]


def build_context(query: str = "", path: str = "", max_files: int = 6, max_chars: int = DEFAULT_CONTEXT_CHARS) -> dict:
    max_files = max(1, min(max_files, 12))
    max_chars = max(1000, min(max_chars, 12000))
    chunks: list[str] = []
    files: list[dict] = []

    if path:
        target = _resolve_repo_path(path)
        files.append(_file_meta(target))
        chunks.append(_context_chunk(target, max_chars))
    else:
        candidates = search_codebase(query, max_files=max_files) if query else build_manifest(limit=max_files)["files"]
        candidate_count = max(1, min(len(candidates), max_files))
        per_file_chars = max(900, min(3000, max_chars // candidate_count))
        for item in candidates[:max_files]:
            target = _resolve_repo_path(item["path"])
            remaining = max_chars - sum(len(chunk) for chunk in chunks)
            if remaining <= 600:
                break
            files.append(_file_meta(target))
            chunks.append(_context_chunk(target, min(per_file_chars, remaining), item.get("matches")))

    context = "\n\n".join(chunks)
    return {
        "root": str(PROJECT_ROOT),
        "query": query,
        "files": files,
        "context": context[:max_chars],
        "policy": "read-only source/docs snippets; secrets, vaults, models, assets, and hidden files are excluded/redacted",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "weaver-codebase-api", "root": str(PROJECT_ROOT)}


@app.get("/manifest")
async def manifest(limit: int = Query(200, ge=1, le=1000)):
    return build_manifest(limit=limit)


@app.get("/file")
async def file(path: str, max_chars: int = Query(DEFAULT_CONTEXT_CHARS, ge=1000, le=12000)):
    target = _resolve_repo_path(path)
    return {**_file_meta(target), "content": _read_text(target, max_chars)}


@app.get("/search")
async def search(query: str, max_files: int = Query(10, ge=1, le=50)):
    return {"query": query, "results": search_codebase(query, max_files=max_files)}


@app.get("/context")
async def context(
    query: str = "",
    path: str = "",
    max_files: int = Query(6, ge=1, le=12),
    max_chars: int = Query(DEFAULT_CONTEXT_CHARS, ge=1000, le=12000),
):
    return build_context(query=query, path=path, max_files=max_files, max_chars=max_chars)


@app.get("/internet/fetch")
async def internet_fetch(
    url: str,
    max_chars: int = Query(6000, ge=1000, le=12000),
):
    return fetch_public_page(url, max_chars=max_chars)


@app.get("/internet/search")
async def internet_search(
    query: str,
    max_results: int = Query(5, ge=1, le=8),
):
    return {"query": query, "results": search_public_web(query, max_results=max_results)}


@app.get("/internet/context")
async def internet_context(
    query: str = "",
    url: str = "",
    max_results: int = Query(4, ge=1, le=6),
    max_chars: int = Query(6000, ge=1000, le=12000),
):
    if url:
        page = fetch_public_page(url, max_chars=max_chars)
        return {
            "query": query,
            "files": [],
            "context": f"### {page['url']}\nTitle: {page['title']}\nPolicy: {page['policy']}\n\n{page['text']}"[:max_chars],
            "policy": page["policy"],
            "results": [{"title": page["title"], "url": page["url"]}],
        }

    results = search_public_web(query, max_results=max_results)
    chunks = []
    budget = max_chars
    for item in results:
        remaining = budget - sum(len(chunk) for chunk in chunks)
        if remaining <= 800:
            break
        try:
            page = fetch_public_page(item["url"], max_chars=min(2200, remaining))
            chunks.append(f"### {page['url']}\nTitle: {page['title'] or item['title']}\n\n{page['text']}")
        except HTTPException:
            chunks.append(f"### {item['url']}\nTitle: {item['title']}\n\n{item['snippet']}")
    return {
        "query": query,
        "results": results,
        "context": "\n\n".join(chunks)[:max_chars],
        "policy": "read-only public internet context; no private/local/reserved hosts, credentials, binary content, or oversized responses",
    }


@app.post("/internet/context")
async def internet_context_post(payload: dict = Body(default_factory=dict)):
    def _int_body(name: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(payload.get(name) or default)))
        except (TypeError, ValueError):
            return default

    max_results = _int_body("max_results", 4, 1, 6)
    max_chars = _int_body("max_chars", 6000, 1000, 12000)
    return await internet_context(
        query=str(payload.get("query") or ""),
        url=str(payload.get("url") or ""),
        max_results=max_results,
        max_chars=max_chars,
    )


async def codebase_api_serve(host: str = "127.0.0.1", port: int = 8091) -> None:
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8091)
