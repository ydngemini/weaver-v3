#!/usr/bin/env python3
"""Seed agentic SWE/tool-calling SFT traces for Weaver MoE distillation.

Each trace follows the 4-step recovery loop:
  1. context injection (ambiguous failure)
  2. environmental probe (discovery tool)
  3. the trap (first call fails: missing context / deprecated var / hw limit)
  4. self-healing (parse raw error, pivot, resolve)

Tool calls use the target SFT syntax:  <|call:tool_name|> {"arg": "val"}
Run:  python gen_agentic_traces.py  ->  agentic_traces.json
"""
import json, os

CALL = lambda name, args: f'<|call:{name}|> ' + json.dumps(args, separators=(", ", ": "))

TRACES = [
    {
        "trace_id": "a1f3c9e2-0001-4b7a-9d11-distdb-schemadrift",
        "domain": "Distributed Databases",
        "prompt": "ALERT: Writes to `orders` succeed on shard-1 but reads return rows with NULL `tenant_id` that were non-null at insert. No migration was deployed in the last 24h. Investigate and restore read integrity.",
        "completion": (
            "Reasoning: NULLs appearing post-insert without a deploy points to a silent schema mutation or a "
            "replica diverging from the primary. I should not assume — discover the live schema per node before "
            "touching data.\n"
            + CALL("query_graph_db", {"query": "SHOW COLUMNS FROM orders", "scope": "all_shards"}) + "\n"
            "System Response: [ERROR] driver=cqlsh code=2200 \"keyspace 'orders' not found on coordinator; "
            "did you mean 'orders_v2'? (alias resolution disabled since release 8.4)\"\n"
            "Recovery reasoning: The alias was deprecated — the table was renamed to `orders_v2` and a compatibility "
            "VIEW masks it for writes but the read path resolves the stale view definition. I must inspect the view, "
            "not the table, then re-point reads.\n"
            + CALL("query_graph_db", {"query": "SHOW CREATE VIEW orders", "scope": "coordinator"}) + "\n"
            "System Response: view orders AS SELECT id, sku, qty, NULL AS tenant_id FROM orders_v2  -- column dropped in mig 0091\n"
            "Recovery reasoning: Confirmed — migration 0091 dropped `tenant_id` from the view projection while the "
            "base table still has it. The fix is to roll the view back to project the real column; no data was lost.\n"
            + CALL("rollback_migration", {"id": "0091", "target": "view:orders", "mode": "projection_only", "dry_run": False}) + "\n"
            "System Response: OK view orders rebuilt; tenant_id now projected from orders_v2.tenant_id (rows affected: 0, definition-only)\n"
            "Resolution: Reads now return non-null tenant_id. Root cause: deprecated alias + lossy view projection in mig 0091, not data corruption. Filed a guard: CI now diffs view projections against base columns."
        ),
    },
    {
        "trace_id": "b2e4da13-0002-4c81-8a22-k8s-crashloop",
        "domain": "Container Orchestration",
        "prompt": "Pod `payments-api` is in CrashLoopBackOff across 3 of 5 replicas after an autoscale event. Healthy replicas predate the scale-up. No image change. Restore full capacity.",
        "completion": (
            "Reasoning: Only the newly-scheduled replicas crash — that smells like node-local context the old pods "
            "already had (a mounted secret, a node label, or a resource ceiling). Probe the cluster state first.\n"
            + CALL("inspect_k8s_cluster", {"resource": "pod", "selector": "app=payments-api", "fields": "status,node,events"}) + "\n"
            "System Response: [FAIL] 3 pods Pending->CrashLoopBackOff; last state: OOMKilled (exit 137); "
            "node pool 'spot-arm64' memory.limit=512Mi; image arch=amd64\n"
            "Recovery reasoning: Two stacked traps: (1) the autoscaler placed pods on a new arm64 spot pool while the "
            "image is amd64 — emulation inflates memory and trips the 512Mi limit -> OOM exit 137. The old replicas "
            "live on the amd64 pool. Pin scheduling to the matching arch and lift the limit.\n"
            + CALL("inspect_k8s_cluster", {"resource": "node", "selector": "kubernetes.io/arch", "fields": "name,arch,allocatable"}) + "\n"
            "System Response: amd64 pool 'std-x86' allocatable mem=4Gi (3 nodes); arm64 pool 'spot-arm64' (2 nodes)\n"
            "Recovery reasoning: Confirmed arch split. Patch the deployment with a nodeAffinity for amd64 and raise the "
            "memory limit to a realistic 768Mi so the autoscaler stops mis-placing.\n"
            + CALL("rollback_migration", {"id": "deploy/payments-api", "target": "spec.template", "patch": {"nodeAffinity": "kubernetes.io/arch=amd64", "resources.limits.memory": "768Mi"}, "dry_run": False}) + "\n"
            "System Response: deployment.apps/payments-api patched; 5/5 replicas Running on std-x86\n"
            "Resolution: Full capacity restored. Root cause was arch-mismatched scheduling under emulation, surfaced only by the scale-up onto the arm64 spot pool — not the image."
        ),
    },
    {
        "trace_id": "c3a59f04-0003-41d9-bb33-locks-deadlock",
        "domain": "Distributed Locks",
        "prompt": "The nightly reconciliation job hangs indefinitely. It acquires per-account advisory locks. Two workers appear stuck. Diagnose and clear without corrupting in-flight reconciliations.",
        "completion": (
            "Reasoning: Two workers stuck on per-account locks is a classic ordered-acquisition deadlock (A holds 17 "
            "wants 42, B holds 42 wants 17). I must observe the wait-for graph before killing anything blindly — a "
            "naive kill could leave a half-applied ledger entry.\n"
            + CALL("query_graph_db", {"query": "SELECT pid, holds, waits_for FROM pg_advisory_wait_graph", "scope": "primary"}) + "\n"
            "System Response: [ERROR] relation \"pg_advisory_wait_graph\" does not exist (custom view removed in db image 14->16 upgrade)\n"
            "Recovery reasoning: The convenience view was dropped in the Postgres 16 upgrade. Pivot to the native "
            "catalogs (pg_locks join pg_stat_activity) to reconstruct the wait-for graph myself.\n"
            + CALL("query_graph_db", {"query": "SELECT a.pid, a.query, l.locktype, l.objid, l.granted FROM pg_locks l JOIN pg_stat_activity a USING (pid) WHERE l.locktype='advisory' ORDER BY a.pid", "scope": "primary"}) + "\n"
            "System Response: pid 8801 granted objid=17, waiting objid=42 | pid 8802 granted objid=42, waiting objid=17\n"
            "Recovery reasoning: Deadlock confirmed and it is symmetric. Killing the lower-progress worker is safe: "
            "the reconciliation job is idempotent per account (it re-derives from the journal). Terminate pid 8802 — "
            "it has done less work this cycle — so 8801 proceeds and 8802 retries cleanly.\n"
            + CALL("rollback_migration", {"id": "pg:terminate", "target": "pid:8802", "mode": "cancel_backend", "dry_run": False}) + "\n"
            "System Response: pg_terminate_backend(8802)=true; pid 8801 acquired objid=42; job progressing\n"
            "Resolution: Reconciliation completed. Permanent fix queued: enforce global lock ordering (always acquire min(account_id) first) to make the deadlock structurally impossible."
        ),
    },
    {
        "trace_id": "d4b60a15-0004-4e2c-9c44-mq-poisonmsg",
        "domain": "Message Queues",
        "prompt": "Consumer group `invoice-emitter` lag is climbing linearly; throughput dropped to zero 12 minutes ago but no consumer crashed. A latent race is suspected. Recover delivery.",
        "completion": (
            "Reasoning: Zero throughput with live consumers and rising lag = head-of-line blocking, usually a poison "
            "message that throws on deserialize and is retried forever at offset N. Probe the consumer group's "
            "current offset and the message at the stall point.\n"
            + CALL("inspect_k8s_cluster", {"resource": "kafka_consumer_group", "selector": "invoice-emitter", "fields": "partition,current_offset,lag"}) + "\n"
            "System Response: partition 3 current_offset=994120 lag=51022 (other partitions lag~0); consumer redelivering offset 994120 x1180 attempts\n"
            "Recovery reasoning: Partition 3 is wedged on offset 994120. Before skipping it I must see WHY it fails — "
            "blind-skipping could drop a valid invoice. Read the raw payload + the consumer error.\n"
            + CALL("read_akashic_state", {"topic": "invoices", "partition": 3, "offset": 994120, "include": "payload,last_error"}) + "\n"
            "System Response: [ERROR] last_error=JsonParse: unexpected token at col 0; payload starts with 0x1f8b (gzip magic); "
            "producer began compressing in release 5.2 but this consumer group predates the codec config\n"
            "Recovery reasoning: Not a poison message — a deprecated consumer config. The producer switched to gzip; "
            "this consumer never got the codec. The fix is to enable decompression and resume from the same offset, NOT skip it.\n"
            + CALL("rollback_migration", {"id": "consumer:invoice-emitter", "target": "config", "patch": {"compression.codec": "gzip", "resume_offset": 994120}, "dry_run": False}) + "\n"
            "System Response: config applied; partition 3 draining; lag 51022->0 over 90s\n"
            "Resolution: Delivery restored with zero data loss — the 'race' was a producer/consumer codec skew introduced in 5.2. Added a contract test asserting consumer codecs match producer settings."
        ),
    },
    {
        "trace_id": "e5c71b26-0005-4f3d-8d55-mesh-mtls",
        "domain": "Service Mesh",
        "prompt": "After a control-plane upgrade, `checkout` -> `inventory` calls return 503 UC (upstream connect). Other routes are fine. mTLS is enabled mesh-wide. Restore the route.",
        "completion": (
            "Reasoning: A single route failing 503 UC after a control-plane upgrade, with mesh-wide mTLS, usually means "
            "a cert/SAN or PeerAuthentication mismatch isolated to one workload pair. Probe the sidecar config dump "
            "for the inventory cluster.\n"
            + CALL("inspect_k8s_cluster", {"resource": "envoy_config", "selector": "app=checkout", "fields": "clusters.inventory.tls_context"}) + "\n"
            "System Response: [WARN] cluster inventory.svc: tls validation_context = SPIFFE trust domain 'cluster.local'; "
            "peer presented 'mesh.internal'; handshake_failure\n"
            "Recovery reasoning: The trap: the upgrade migrated the trust domain to 'mesh.internal' but inventory's "
            "PeerAuthentication still pins the old 'cluster.local'. Confirm which workloads are on which domain before "
            "flipping anything.\n"
            + CALL("query_graph_db", {"query": "SELECT workload, trust_domain, mode FROM peer_authentication", "scope": "mesh"}) + "\n"
            "System Response: checkout->mesh.internal STRICT | inventory->cluster.local STRICT (not migrated)\n"
            "Recovery reasoning: inventory was missed in the trust-domain rollout. Re-issue its identity under "
            "'mesh.internal' and apply matching PeerAuthentication, keeping STRICT so we never downgrade security.\n"
            + CALL("rollback_migration", {"id": "peerauth/inventory", "target": "spec", "patch": {"trust_domain": "mesh.internal", "mode": "STRICT"}, "dry_run": False}) + "\n"
            "System Response: identity re-issued; checkout->inventory 200 OK; 503 UC cleared\n"
            "Resolution: Route healthy. Cause was a partial trust-domain migration during the control-plane upgrade — fix preserved STRICT mTLS throughout."
        ),
    },
    {
        "trace_id": "f6d82c37-0006-4a4e-9e66-cache-stampede",
        "domain": "Caching & State",
        "prompt": "p99 latency on `/profile` spiked 40x at the top of the hour and the DB is at 100% CPU, though traffic is flat. Suspected cache stampede or stale-read race. Stabilize.",
        "completion": (
            "Reasoning: 40x p99 with flat traffic + DB pegged at a round time boundary screams synchronized cache "
            "expiry -> stampede: thousands of keys TTL out at :00 and all miss to the DB at once. Probe the cache key "
            "TTL distribution to confirm.\n"
            + CALL("read_akashic_state", {"store": "redis", "pattern": "profile:*", "include": "ttl_histogram"}) + "\n"
            "System Response: [ERROR] CROSSSLOT keys in request don't hash to the same slot; SCAN with MATCH across "
            "16384 slots refused on cluster mode (deprecated single-node assumption)\n"
            "Recovery reasoning: The probe assumed single-node Redis; the store is now clustered. Pivot to a per-node "
            "SCAN fan-out to gather the TTL histogram without a cross-slot op.\n"
            + CALL("read_akashic_state", {"store": "redis_cluster", "pattern": "profile:*", "include": "ttl_histogram", "fanout": "per_node"}) + "\n"
            "System Response: 92% of profile:* keys share ttl expiry epoch=1718446800 (exactly :00); no jitter\n"
            "Recovery reasoning: Confirmed: zero TTL jitter -> mass synchronized expiry. Two-part fix: add randomized "
            "TTL jitter going forward and, right now, enable request coalescing (single-flight) so only one loader per "
            "key hits the DB while it warms.\n"
            + CALL("rollback_migration", {"id": "cache:profile", "target": "policy", "patch": {"ttl_jitter_pct": 20, "single_flight": True}, "dry_run": False}) + "\n"
            "System Response: policy applied; DB CPU 100%->34% in 45s; /profile p99 back to baseline\n"
            "Resolution: Stabilized. Root cause was jitter-free TTLs causing a top-of-hour stampede, masked initially by a deprecated single-node cache assumption in tooling."
        ),
    },
    {
        "trace_id": "07e93d48-0007-4b5f-8f77-akashic-statecorrupt",
        "domain": "Agentic State (Weaver Akashic Hub)",
        "prompt": "The Nexus Bus reports STATUS_UPDATE storms and the Akashic Hub returns conflicting lobe states on consecutive reads. Restore a consistent state snapshot without losing recent observations.",
        "completion": (
            "Reasoning: Conflicting consecutive reads from the Akashic Hub points to a split between the in-memory hub "
            "and the on-disk restore (the hub rehydrates from disk on startup). A storm of STATUS_UPDATE suggests two "
            "writers. Probe the hub's state provenance first.\n"
            + CALL("read_akashic_state", {"scope": "hub", "include": "epoch,writer_id,disk_checksum,mem_checksum"}) + "\n"
            "System Response: [ERROR] mem_checksum != disk_checksum; two writer_ids active (pid 4120 epoch=8, pid 4188 epoch=8); "
            "WS bind on 9999 held by both (SO_REUSEPORT left enabled after port-cleanup patch)\n"
            "Recovery reasoning: The trap is a duplicate hub process — a stale instance survived the port-cleanup and "
            "both bound 9999 via SO_REUSEPORT, so observations split across two epochs. I must identify the authoritative "
            "writer (highest committed offset) before reaping the other.\n"
            + CALL("inspect_k8s_cluster", {"resource": "process", "selector": "nexus-hub", "fields": "pid,committed_offset,started_at"}) + "\n"
            "System Response: pid 4120 committed_offset=559210 started 02:11 | pid 4188 committed_offset=559730 started 09:42\n"
            "Recovery reasoning: pid 4188 is newer and ahead by 520 observations — make it authoritative, merge 4120's "
            "unique tail, then terminate the stale 4120 and re-snapshot disk from the merged memory state so a restart "
            "rehydrates cleanly.\n"
            + CALL("rollback_migration", {"id": "akashic:reconcile", "target": "hub", "patch": {"authoritative_pid": 4188, "merge_from": 4120, "reap_stale": True, "resnapshot_disk": True}, "dry_run": False}) + "\n"
            "System Response: merged 41 unique observations from 4120; pid 4120 reaped; disk snapshot epoch=9 checksum matched mem; bus quiet\n"
            "Resolution: Single authoritative hub, consistent reads, zero observation loss. Permanent fix: disable SO_REUSEPORT on the hub socket so a stale process can't co-bind 9999."
        ),
    },
    {
        "trace_id": "18fa4e59-0008-4c6a-9088-async-deprecatedenv",
        "domain": "Async Pipelines",
        "prompt": "An asyncio ingestion worker silently processes 0 documents but exits 0 (green in CI). Downstream notices missing data hours later. Find why it no-ops and fix it.",
        "completion": (
            "Reasoning: Exit 0 + zero work + green CI is the most dangerous failure: a swallowed exception or a config "
            "that resolves to an empty work set. Probe the worker's resolved runtime config and the document cursor.\n"
            + CALL("read_akashic_state", {"component": "ingest_worker", "include": "resolved_env,cursor,last_batch_size"}) + "\n"
            "System Response: [WARN] resolved_env: SOURCE_URI=None (key 'INGEST_SOURCE_URI' read; deploy sets "
            "'INGEST_SOURCE_URL'); cursor=0; last_batch_size=0; no exception raised\n"
            "Recovery reasoning: The trap: a deprecated env var name. Code reads INGEST_SOURCE_URI; the deploy template "
            "renamed it to ..._URL. None -> empty source -> the async generator yields nothing -> clean exit 0. I must "
            "verify the live deploy key before patching so I fix the real skew.\n"
            + CALL("inspect_k8s_cluster", {"resource": "deployment", "selector": "ingest-worker", "fields": "env"}) + "\n"
            "System Response: env: INGEST_SOURCE_URL=s3://corpus/2026/ ; (INGEST_SOURCE_URI absent)\n"
            "Recovery reasoning: Confirmed. Two fixes: (1) make the code accept both keys with URL canonical, and (2) "
            "add a startup assertion so an empty source fails loud (non-zero) instead of silent no-op. Re-run the batch "
            "after patch.\n"
            + CALL("rollback_migration", {"id": "ingest_worker", "target": "config_loader", "patch": {"alias": {"INGEST_SOURCE_URI": "INGEST_SOURCE_URL"}, "fail_on_empty_source": True, "rerun_batch": True}, "dry_run": False}) + "\n"
            "System Response: source resolved s3://corpus/2026/; processed 48211 docs; assertion guard active\n"
            "Resolution: Backfill complete. The silent no-op was a deprecated env-var name; the guard converts that failure class from silent to loud forever."
        ),
    },
    {
        "trace_id": "29ab5f6a-0009-4d7b-9199-storage-diskceiling",
        "domain": "Storage & Hardware Limits",
        "prompt": "A long-running data-prep job crashes near completion with a generic IOError after consuming ~6B tokens; the output file is 0 bytes despite hours of work. Recover without re-running from scratch.",
        "completion": (
            "Reasoning: 0-byte output after near-complete work + generic IOError almost always means the final atomic "
            "write/rename failed — and the usual culprit is the disk filling exactly at finalize. Probe the volume and "
            "the job's checkpoint cursor before assuming the data is gone.\n"
            + CALL("inspect_k8s_cluster", {"resource": "volume", "selector": "job=data-prep", "fields": "mount,used_pct,inodes"}) + "\n"
            "System Response: [ERROR] /kaggle/working used_pct=99% (free=410MB); ENOSPC during os.replace(tmp->train.bin); "
            "tmp shard train.bin.partial=11.4GB present\n"
            "Recovery reasoning: The trap is a hardware/disk ceiling, not lost data: the 11.4GB partial shard survived; "
            "only the final atomic rename ran out of space. The per-substream cursor should let me finalize without "
            "re-tokenizing. Confirm the cursor covers the corpus.\n"
            + CALL("read_akashic_state", {"job": "data-prep", "include": "cursor,docs_consumed,expected_tokens"}) + "\n"
            "System Response: cursor: all substreams exhausted; docs_consumed=919117; tokens_written=6000002701 (matches target)\n"
            "Recovery reasoning: All tokens are physically in the partial shard; the job is data-complete. I need to "
            "free headroom (drop obsolete backups), then finalize from the partial rather than recompute, and write to "
            "the larger partition.\n"
            + CALL("rollback_migration", {"id": "data-prep:finalize", "target": "output", "patch": {"free_space": ["drop:old_2.29B_backup"], "finalize_from": "train.bin.partial", "dest": "/project/recover_v5/train.bin"}, "dry_run": False}) + "\n"
            "System Response: freed 4.3GB; renamed partial->train.bin (11.4GB) on /project; meta.json finished=true tokens=6000002701\n"
            "Resolution: Recovered the full 6B-token dataset with zero recompute. Root cause: 99% disk at finalize-time ENOSPC. Guard added: pre-finalize free-space check >= 1.5x tmp shard size."
        ),
    },
    {
        "trace_id": "3abc607b-0010-4e8c-92aa-graphdb-cycledep",
        "domain": "Dependency Graphs",
        "prompt": "A monorepo build deadlocks: target `service-a` waits on `lib-x` which (transitively) waits on `service-a`. The build tool just reports a timeout. Break the cycle correctly.",
        "completion": (
            "Reasoning: A build timeout with mutual waits is a dependency cycle. I should not hand-guess the edges — "
            "extract the actual dependency graph and find the strongly-connected component.\n"
            + CALL("query_graph_db", {"query": "MATCH path=(n)-[:DEPENDS_ON*]->(n) RETURN nodes(path) LIMIT 5", "scope": "build_graph"}) + "\n"
            "System Response: [ERROR] query rejected: variable-length pattern exceeds max_hops=3 (lowered from 15 in graph engine 3.0 for safety)\n"
            "Recovery reasoning: The trap: the engine upgrade capped variable-length traversal at 3 hops, so my cycle "
            "query is refused. Pivot to the engine's native SCC algorithm instead of an unbounded path match.\n"
            + CALL("query_graph_db", {"query": "CALL gds.scc.stream('build_graph') YIELD nodeId, componentId WITH componentId, collect(nodeId) AS c WHERE size(c) > 1 RETURN c", "scope": "build_graph"}) + "\n"
            "System Response: component: [service-a, lib-x, lib-y]  (cycle: service-a->lib-x->lib-y->service-a)\n"
            "Recovery reasoning: The real cycle runs through lib-y. service-a depends on lib-y only for a single type "
            "definition — the correct fix is to extract that interface into a leaf package `lib-y-types` so the cycle "
            "breaks without merging modules. Apply the edge rewrite.\n"
            + CALL("rollback_migration", {"id": "build_graph", "target": "edges", "patch": {"extract": "lib-y::types -> lib-y-types", "repoint": "service-a:DEPENDS_ON->lib-y-types", "remove": "lib-y-types:DEPENDS_ON->service-a"}, "dry_run": False}) + "\n"
            "System Response: graph acyclic (SCC sizes all=1); build scheduled; service-a + lib-x + lib-y compiled\n"
            "Resolution: Build green. Cycle broken by extracting a leaf types package, not by force-merging. Root probe failure was a post-upgrade max_hops cap — switched cycle detection to the native SCC procedure."
        ),
    },
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_traces.json")
with open(OUT, "w") as f:
    json.dump(TRACES, f, indent=2, ensure_ascii=False)
print(f"wrote {len(TRACES)} traces -> {OUT}")
