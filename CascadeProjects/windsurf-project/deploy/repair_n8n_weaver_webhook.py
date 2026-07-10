#!/usr/bin/env python3
"""Repair Weaver's persisted n8n webhook route after workflow restore/import."""

from __future__ import annotations

import argparse
import contextlib
import os
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
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "no such object" in detail.lower() or "no such container" in detail.lower():
            return False
        raise RuntimeError(detail or f"failed to inspect {container}")
    return result.stdout.strip().lower() == "true"


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


def _backup_database(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = db_path.with_name(f"{db_path.name}.backup.{stamp}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source:
        with sqlite3.connect(str(backup)) as destination:
            source.backup(destination)
            integrity = destination.execute("pragma integrity_check").fetchone()[0]
    if integrity != "ok" or backup.stat().st_size == 0:
        backup.unlink(missing_ok=True)
        raise RuntimeError(f"n8n backup verification failed: {integrity}")
    source_stat = db_path.stat()
    os.chmod(backup, 0o600)
    with contextlib.suppress(PermissionError):
        os.chown(backup, source_stat.st_uid, source_stat.st_gid)
    backups = sorted(
        db_path.parent.glob(f"{db_path.name}.backup.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in backups[5:]:
        stale.unlink()
    return backup


def repair(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(f"n8n database not found: {db_path}", file=sys.stderr)
        return 1

    restarted = False
    try:
        if not args.container and not args.offline:
            raise RuntimeError("refusing live/offline-unknown repair; pass --offline after stopping n8n")
        if args.no_container_restart and not args.offline:
            raise RuntimeError("--no-container-restart requires --offline")
        if not args.no_container_restart and args.container:
            restarted = _stop_container(args.container)

        if not args.no_backup:
            backup = _backup_database(db_path)
            print(f"backup={backup}")
        if args.backup_only:
            return 0

        with sqlite3.connect(str(db_path), timeout=30) as con:
            con.execute("pragma foreign_keys=on")
            con.execute("begin immediate")
            tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
            if "workflow_entity" not in tables:
                raise RuntimeError("workflow_entity table is missing")
            workflow_columns = {row[1] for row in con.execute("pragma table_info(workflow_entity)")}
            select_columns = "id, active" if "active" in workflow_columns else "id"
            workflow = con.execute(
                f"select {select_columns} from workflow_entity where id=?",
                (args.workflow_id,),
            ).fetchone()
            if workflow is None:
                raise RuntimeError(f"workflow {args.workflow_id!r} does not exist")
            if len(workflow) > 1 and not bool(workflow[1]):
                raise RuntimeError(f"workflow {args.workflow_id!r} is not active")

            conflicts = con.execute(
                "select workflowId, node from webhook_entity where webhookPath=? and method=?",
                (args.webhook_path, args.method),
            ).fetchall()
            foreign = [row for row in conflicts if row[0] != args.workflow_id or row[1] != args.node]
            if foreign:
                raise RuntimeError(
                    f"webhook route {args.method} {args.webhook_path!r} is owned by {foreign}"
                )

            columns = {row[1] for row in con.execute("pragma table_info(webhook_entity)")}
            values = {
                "workflowId": args.workflow_id,
                "webhookPath": args.webhook_path,
                "method": args.method,
                "node": args.node,
                "webhookId": args.webhook_id,
                "pathLength": _path_length(args.webhook_path),
            }
            required = {"workflowId", "webhookPath", "method", "node"}
            if not required.issubset(columns):
                raise RuntimeError(f"unsupported webhook_entity schema: {sorted(columns)}")

            con.execute(
                "delete from webhook_entity where workflowId=? and node=? and method=?",
                (args.workflow_id, args.node, args.method),
            )
            use = {key: value for key, value in values.items() if key in columns}
            con.execute(
                f"insert into webhook_entity ({','.join(use)}) values ({','.join('?' for _ in use)})",
                list(use.values()),
            )
            verify_columns = ["workflowId", "webhookPath", "method", "node"]
            expected = [args.workflow_id, args.webhook_path, args.method, args.node]
            for optional, value in (("webhookId", args.webhook_id), ("pathLength", _path_length(args.webhook_path))):
                if optional in columns:
                    verify_columns.append(optional)
                    expected.append(value)
            row = con.execute(
                f"select {','.join(verify_columns)} from webhook_entity "
                "where webhookPath=? and method=?",
                (args.webhook_path, args.method),
            ).fetchone()
            if row != tuple(expected):
                raise RuntimeError(f"webhook verification failed: {row!r}")
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
    parser.add_argument(
        "--container",
        default="",
        help="optional Docker container to stop/start; omit for systemd-managed n8n",
    )
    parser.add_argument("--no-container-restart", action="store_true", help="patch without stopping/starting Docker")
    parser.add_argument("--offline", action="store_true", help="confirm n8n is already stopped")
    parser.add_argument("--no-backup", action="store_true", help="skip timestamped database backup")
    parser.add_argument("--backup-only", action="store_true", help="create and verify a backup without mutation")
    parser.add_argument("--workflow-id", default="weaverv5soulbind")
    parser.add_argument("--node", default="1. Input Gateway")
    parser.add_argument("--method", default="POST")
    parser.add_argument("--webhook-id", default="weaver-input")
    parser.add_argument("--webhook-path", default="weaver-input")
    return repair(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
