#!/usr/bin/env python3
"""Repair Weaver's persisted n8n webhook route after workflow restore/import."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB = "/var/lib/docker/volumes/n8n_data/_data/database.sqlite"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def _container_running(container: str) -> bool:
    if not container:
        return False
    result = _run(["docker", "inspect", "-f", "{{.State.Running}}", container])
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _stop_container(container: str) -> bool:
    if not _container_running(container):
        return False
    result = _run(["docker", "stop", container])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to stop {container}")
    return True


def _start_container(container: str) -> None:
    result = _run(["docker", "start", container])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to start {container}")


def _path_length(webhook_path: str) -> int:
    return len([part for part in webhook_path.split("/") if part])


def repair(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(f"n8n database not found: {db_path}", file=sys.stderr)
        return 1

    restarted = False
    try:
        if not args.no_container_restart and args.container:
            restarted = _stop_container(args.container)

        if not args.no_backup:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = db_path.with_name(f"{db_path.name}.backup.{stamp}")
            shutil.copy2(db_path, backup)
            print(f"backup={backup}")

        with sqlite3.connect(str(db_path)) as con:
            cur = con.cursor()
            cur.execute(
                """
                update webhook_entity
                   set webhookPath = ?,
                       webhookId = ?,
                       pathLength = ?
                 where workflowId = ?
                   and node = ?
                   and method = ?
                """,
                (
                    args.webhook_path,
                    args.webhook_id,
                    _path_length(args.webhook_path),
                    args.workflow_id,
                    args.node,
                    args.method,
                ),
            )
            if cur.rowcount == 0:
                con.rollback()
                print(
                    "no matching webhook row found for "
                    f"workflow={args.workflow_id!r} node={args.node!r} method={args.method!r}",
                    file=sys.stderr,
                )
                return 2
            con.commit()

        print(
            "repaired n8n webhook: "
            f"workflow={args.workflow_id} node={args.node} method={args.method} "
            f"path={args.webhook_path}"
        )
        return 0
    finally:
        if restarted:
            _start_container(args.container)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair Weaver's n8n production webhook route in database.sqlite."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"n8n SQLite DB path (default: {DEFAULT_DB})")
    parser.add_argument("--container", default="n8n", help="n8n Docker container name")
    parser.add_argument("--no-container-restart", action="store_true", help="patch without stopping/starting Docker")
    parser.add_argument("--no-backup", action="store_true", help="skip timestamped database backup")
    parser.add_argument("--workflow-id", default="weaverv5soulbind")
    parser.add_argument("--node", default="1. Input Gateway")
    parser.add_argument("--method", default="POST")
    parser.add_argument("--webhook-id", default="weaver-input")
    parser.add_argument("--webhook-path", default="weaver-input")
    return repair(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
