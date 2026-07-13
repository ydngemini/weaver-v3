#!/usr/bin/env node
/**
 * Deterministic offline validator for Weaver's production n8n workflow.
 *
 * It validates more than JSON syntax: node/edge integrity, reachability,
 * terminal response paths, cycles, Merge input contracts, named-node
 * expression ancestry, Code-node JavaScript syntax, HTTP allowlists,
 * credential references, retry/deadline budgets, privacy invariants, and the
 * parallel expert/local-model topology.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = resolve(fileURLToPath(new URL('.', import.meta.url)));
const projectRoot = resolve(here, '..');
const args = process.argv.slice(2);
const jsonOutput = args.includes('--json');
const requestedPath = args.find((arg) => !arg.startsWith('--'));
const workflowPath = resolve(requestedPath || resolve(projectRoot, 'n8n_weaver_v5.json'));

const errors = [];
const warnings = [];
const checks = [];

function check(condition, message, detail = undefined) {
  checks.push({ message, ok: Boolean(condition), ...(detail === undefined ? {} : { detail }) });
  if (!condition) errors.push(message);
}

function warn(condition, message) {
  if (!condition) warnings.push(message);
}

let raw = '';
let workflow;
try {
  raw = readFileSync(workflowPath, 'utf8');
  workflow = JSON.parse(raw);
} catch (error) {
  const result = {
    valid: false,
    workflow: workflowPath,
    errors: [`workflow JSON could not be loaded: ${error.message}`],
    warnings: [],
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(1);
}

check(workflow && typeof workflow === 'object' && !Array.isArray(workflow), 'workflow root must be an object');
check(typeof workflow.name === 'string' && workflow.name.length > 0, 'workflow name is required');
check(Array.isArray(workflow.nodes) && workflow.nodes.length > 0, 'workflow nodes must be a non-empty array');
check(workflow.connections && typeof workflow.connections === 'object', 'workflow connections must be an object');
check(workflow.active === true, 'production workflow must be active in the exported contract');

const nodes = new Map();
const ids = new Set();
for (const [index, node] of (workflow.nodes || []).entries()) {
  const prefix = `node[${index}]`;
  check(node && typeof node === 'object' && !Array.isArray(node), `${prefix} must be an object`);
  if (!node || typeof node !== 'object') continue;
  check(typeof node.name === 'string' && node.name.length > 0, `${prefix} name is required`);
  check(typeof node.id === 'string' && node.id.length > 0, `${prefix} id is required`);
  check(typeof node.type === 'string' && node.type.length > 0, `${prefix} type is required`);
  check(Number.isFinite(Number(node.typeVersion)), `${prefix} typeVersion must be numeric`);
  check(Array.isArray(node.position) && node.position.length === 2, `${prefix} position must be [x,y]`);
  if (typeof node.name === 'string') {
    check(!nodes.has(node.name), `duplicate node name: ${node.name}`);
    nodes.set(node.name, node);
  }
  if (typeof node.id === 'string') {
    check(!ids.has(node.id), `duplicate node id: ${node.id}`);
    ids.add(node.id);
  }
}

const adjacency = new Map([...nodes.keys()].map((name) => [name, new Set()]));
const reverse = new Map([...nodes.keys()].map((name) => [name, new Set()]));
const inboundByInput = new Map([...nodes.keys()].map((name) => [name, new Map()]));
let edgeCount = 0;

for (const [source, ports] of Object.entries(workflow.connections || {})) {
  check(nodes.has(source), `connection source does not exist: ${source}`);
  check(ports && typeof ports === 'object' && !Array.isArray(ports), `connection ports invalid: ${source}`);
  if (!nodes.has(source) || !ports || typeof ports !== 'object') continue;
  for (const [portType, branches] of Object.entries(ports)) {
    check(portType === 'main', `unsupported connection port ${portType} on ${source}`);
    check(Array.isArray(branches), `connection branches must be an array: ${source}.${portType}`);
    if (!Array.isArray(branches)) continue;
    for (const [outputIndex, branch] of branches.entries()) {
      check(Array.isArray(branch), `connection branch must be an array: ${source}.${portType}[${outputIndex}]`);
      if (!Array.isArray(branch)) continue;
      for (const connection of branch) {
        const target = connection?.node;
        const inputIndex = connection?.index;
        check(typeof target === 'string' && nodes.has(target), `connection target does not exist: ${source} -> ${String(target)}`);
        check(connection?.type === 'main', `connection type must be main: ${source} -> ${String(target)}`);
        check(Number.isInteger(inputIndex) && inputIndex >= 0, `connection input index invalid: ${source} -> ${String(target)}`);
        if (typeof target !== 'string' || !nodes.has(target) || !Number.isInteger(inputIndex)) continue;
        adjacency.get(source).add(target);
        reverse.get(target).add(source);
        const inputMap = inboundByInput.get(target);
        if (!inputMap.has(inputIndex)) inputMap.set(inputIndex, []);
        inputMap.get(inputIndex).push(source);
        edgeCount += 1;
      }
    }
  }
}

const webhookNodes = [...nodes.values()].filter((node) => node.type === 'n8n-nodes-base.webhook');
check(webhookNodes.length === 1, 'workflow must contain exactly one Webhook node');
const webhook = webhookNodes[0];
if (webhook) {
  check(webhook.parameters?.httpMethod === 'POST', 'Webhook must accept POST only');
  check(webhook.parameters?.path === 'weaver-input', 'Webhook path must be weaver-input');
  check(webhook.parameters?.responseMode === 'lastNode', 'Webhook must respond with the terminal node');
}

const entryName = webhook?.name;
const reachable = new Set();
if (entryName && nodes.has(entryName)) {
  const queue = [entryName];
  while (queue.length) {
    const current = queue.shift();
    if (reachable.has(current)) continue;
    reachable.add(current);
    for (const target of adjacency.get(current) || []) queue.push(target);
  }
}
for (const name of nodes.keys()) {
  check(reachable.has(name), `node is unreachable from Webhook: ${name}`);
}

const terminals = [...reachable].filter((name) => (adjacency.get(name)?.size || 0) === 0);
check(terminals.length === 1, `workflow must have exactly one terminal node, found: ${terminals.join(', ')}`);
check(terminals[0] === '9. Writeback', 'all response branches must terminate at 9. Writeback');

const canReachTerminal = new Set();
if (nodes.has('9. Writeback')) {
  const queue = ['9. Writeback'];
  while (queue.length) {
    const current = queue.shift();
    if (canReachTerminal.has(current)) continue;
    canReachTerminal.add(current);
    for (const source of reverse.get(current) || []) queue.push(source);
  }
}
for (const name of reachable) {
  check(canReachTerminal.has(name), `reachable node has no response path: ${name}`);
}

const visiting = new Set();
const visited = new Set();
function visit(name, trail = []) {
  if (visiting.has(name)) {
    errors.push(`workflow cycle detected: ${[...trail, name].join(' -> ')}`);
    return;
  }
  if (visited.has(name)) return;
  visiting.add(name);
  for (const target of adjacency.get(name) || []) visit(target, [...trail, name]);
  visiting.delete(name);
  visited.add(name);
}
if (entryName) visit(entryName);

function ancestorsOf(name) {
  const found = new Set();
  const queue = [...(reverse.get(name) || [])];
  while (queue.length) {
    const current = queue.shift();
    if (found.has(current)) continue;
    found.add(current);
    for (const source of reverse.get(current) || []) queue.push(source);
  }
  return found;
}

for (const node of nodes.values()) {
  if (node.type !== 'n8n-nodes-base.merge') continue;
  const count = Number(node.parameters?.numberInputs || 2);
  check(Number.isInteger(count) && count >= 2 && count <= 10, `${node.name}: Merge numberInputs must be 2..10`);
  const inputMap = inboundByInput.get(node.name) || new Map();
  for (let index = 0; index < count; index += 1) {
    check((inputMap.get(index) || []).length === 1, `${node.name}: Merge input ${index} must have exactly one source`);
  }
  for (const inputIndex of inputMap.keys()) {
    check(inputIndex < count, `${node.name}: connection targets undeclared Merge input ${inputIndex}`);
  }
}

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
let codeNodeCount = 0;
let expressionCount = 0;
const namedReferencePattern = /\$\(\s*['"]([^'"]+)['"]\s*\)/g;
const forbiddenCodePatterns = [
  [/\brequire\s*\(/, 'require()'],
  [/\bprocess\s*\./, 'process access'],
  [/child_process/, 'child_process'],
  [/\beval\s*\(/, 'eval()'],
  [/\bFunction\s*\(/, 'Function()'],
];

function validateNamedReferences(node, text, location) {
  const ancestors = ancestorsOf(node.name);
  for (const match of text.matchAll(namedReferencePattern)) {
    const target = match[1];
    check(nodes.has(target), `${node.name}: ${location} references missing node ${target}`);
    if (nodes.has(target)) {
      check(ancestors.has(target), `${node.name}: ${location} references non-upstream node ${target}`);
    }
  }
}

function walkExpressions(node, value, path = 'parameters') {
  if (typeof value === 'string') {
    if (value.startsWith('={{') && value.endsWith('}}')) {
      expressionCount += 1;
      const body = value.slice(3, -2).trim();
      try {
        // Compilation only. No expression is executed by this validator.
        new AsyncFunction('$json', '$now', '$execution', '$workflow', '$', `return (${body});`);
      } catch (error) {
        errors.push(`${node.name}: invalid expression syntax at ${path}: ${error.message}`);
      }
      validateNamedReferences(node, body, `expression ${path}`);
    }
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => walkExpressions(node, item, `${path}[${index}]`));
    return;
  }
  if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, item]) => walkExpressions(node, item, `${path}.${key}`));
  }
}

for (const node of nodes.values()) {
  walkExpressions(node, node.parameters || {});
  if (node.type !== 'n8n-nodes-base.code') continue;
  codeNodeCount += 1;
  const code = node.parameters?.jsCode;
  check(typeof code === 'string' && code.length > 0, `${node.name}: Code node jsCode is required`);
  if (typeof code !== 'string') continue;
  try {
    // Compilation only. n8n globals are function parameters and code is never run.
    new AsyncFunction('$input', '$execution', '$workflow', '$now', '$', code);
  } catch (error) {
    errors.push(`${node.name}: invalid JavaScript syntax: ${error.message}`);
  }
  validateNamedReferences(node, code, 'Code node');
  const executableCode = code
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '');
  for (const [pattern, label] of forbiddenCodePatterns) {
    check(!pattern.test(executableCode), `${node.name}: forbidden sandbox capability ${label}`);
  }
}

const allowedExternalHosts = new Set(['bedrock-mantle.us-east-1.api.aws']);
const allowedInternalPorts = new Set(['8091', '8898', '8899']);
const httpNodes = [...nodes.values()].filter((node) => node.type === 'n8n-nodes-base.httpRequest');
for (const node of httpNodes) {
  const parameters = node.parameters || {};
  const rawUrl = parameters.url;
  check(typeof rawUrl === 'string' && !rawUrl.startsWith('='), `${node.name}: HTTP URL must be static`);
  let url;
  try {
    url = new URL(rawUrl);
  } catch {
    errors.push(`${node.name}: HTTP URL is invalid`);
    continue;
  }
  const external = allowedExternalHosts.has(url.hostname);
  const internal = url.hostname === 'host.docker.internal' && allowedInternalPorts.has(url.port);
  check(external || internal, `${node.name}: HTTP host/port is not allowlisted`);
  check((external && url.protocol === 'https:') || (internal && url.protocol === 'http:'), `${node.name}: HTTP scheme is invalid`);
  check(parameters.method === 'POST', `${node.name}: HTTP method must be POST`);
  check(parameters.sendBody === true, `${node.name}: HTTP body must be enabled`);
  check(parameters.specifyBody === 'json', `${node.name}: HTTP body must be JSON`);
  const timeout = Number(parameters.options?.timeout);
  check(Number.isInteger(timeout) && timeout >= 100 && timeout <= 30_000, `${node.name}: timeout must be 100..30000 ms`);
  check(node.continueOnFail === true, `${node.name}: continueOnFail must be enabled`);
  check(node.alwaysOutputData === true, `${node.name}: alwaysOutputData must be enabled`);
  if (external) {
    check(parameters.authentication === 'genericCredentialType', `${node.name}: external call must use a credential type`);
    check(parameters.genericAuthType === 'httpHeaderAuth', `${node.name}: external call must use Header Auth`);
    check(Boolean(node.credentials?.httpHeaderAuth?.id), `${node.name}: Header Auth credential reference is missing`);
  }
}

const serialized = JSON.stringify(workflow);
const secretPatterns = [
  /sk-(?:proj-)?[A-Za-z0-9_-]{16,}/,
  /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,
  /\bBearer\s+[A-Za-z0-9._-]{20,}/i,
  /(?:api[_-]?key|password|secret)\s*[:=]\s*['"][^'"]{12,}['"]/i,
];
for (const pattern of secretPatterns) {
  check(!pattern.test(serialized), `workflow appears to contain a hardcoded secret matching ${pattern}`);
}

const requiredNodes = [
  '1. Input Gateway', '2. Sanitize', '3. Error Gate', 'DLQ Logger',
  '4. Fracture+Gate', '5. Expert Fanout', '5f. Expert Barrier',
  '5g. Expert Assembly', '5a. Logic', '5b. Emotion', '5c. Memory',
  '5d. Creativity', '5e. Vigilance', 'Internet Context Bridge',
  '6. Collapse', '7. Self-Reflect', '8. LoRA Voice', '8b. Qwen3B',
  '8c. Local Barrier', '8c. Dual Merge', '9. Writeback',
];
for (const name of requiredNodes) check(nodes.has(name), `required production node is missing: ${name}`);

const expertTargets = new Set(adjacency.get('5. Expert Fanout') || []);
const expectedExperts = new Set(['5a. Logic', '5b. Emotion', '5c. Memory', '5d. Creativity', '5e. Vigilance']);
check(
  expertTargets.size === expectedExperts.size && [...expectedExperts].every((name) => expertTargets.has(name)),
  'Expert Fanout must dispatch exactly the five expert lobes in parallel',
);
const expertBarrierInputs = inboundByInput.get('5f. Expert Barrier') || new Map();
check(expertBarrierInputs.size === 5, 'Expert Barrier must synchronize all five expert tags');
const localTargets = new Set(adjacency.get('7-tag') || []);
check(localTargets.has('8. LoRA Voice') && localTargets.has('8b. Qwen3B') && localTargets.size === 2, 'Self-reflection must fan out to both local models');
check((inboundByInput.get('8c. Local Barrier') || new Map()).size === 2, 'Local Barrier must synchronize both local-model tags');

const sanitizeCode = nodes.get('2. Sanitize')?.parameters?.jsCode || '';
const n8nContractVersion = 'weaver-headless-n8n-v1';
const requestContractFields = [
  'contract_version', 'correlation_id', 'deadline_ms', 'text',
  'self_check', 'introspect', 'path_glob', 'search_query',
  'codebase_context', 'quantum_pathway', 'cognition_context',
];
check(sanitizeCode.includes(n8nContractVersion), 'Sanitize must require the versioned headless contract');
for (const field of requestContractFields) {
  check(sanitizeCode.includes(`'${field}'`), `Sanitize contract is missing request field ${field}`);
}
check(
  sanitizeCode.includes('keys.length === ALLOWED.length')
    && sanitizeCode.includes('keys.every(key => ALLOWED.includes(key))'),
  'Sanitize must reject missing and additional top-level fields',
);
check(sanitizeCode.includes('body.text.trim().length <= 4000'), 'Sanitize must cap user text at 4000 characters');
check(sanitizeCode.includes('body.codebase_context.length <= 12000'), 'Sanitize must cap codebase evidence at 12000 characters');
check(sanitizeCode.includes('body.deadline_ms === 115000'), 'Sanitize must enforce the 115-second brain deadline');
check(sanitizeCode.includes('body.correlation_id === correlation'), 'Sanitize must preserve the authenticated correlation ID');
check(sanitizeCode.includes('cognitionKeys.length === COGNITION_ALLOWED.length'), 'Sanitize must enforce the exact Cognition Mesh context');
check(!sanitizeCode.includes('body.selfCheck'), 'Sanitize must not accept compatibility aliases outside the v1 contract');
const dlqCode = nodes.get('DLQ Logger')?.parameters?.jsCode || '';
check(dlqCode.includes('METADATA-ONLY'), 'DLQ must declare metadata-only handling');
check(!dlqCode.includes('appendFile') && !dlqCode.includes('dlq_path'), 'DLQ must not persist raw request data from a Code node');
check(dlqCode.includes("error_code: 'invalid-request'"), 'DLQ must produce the stable invalid-request rejection');
check(!dlqCode.includes('...d') && !dlqCode.includes('...safe'), 'DLQ must not spread request fields into its envelope');
const writebackCode = nodes.get('9. Writeback')?.parameters?.jsCode || '';
const speakerBody = nodes.get('7. Self-Reflect')?.parameters?.jsonBody || '';
const speakerTagCode = nodes.get('7-tag')?.parameters?.jsCode || '';
check(speakerBody.includes('qwen.qwen3-235b-a22b-2507'), 'Public speaker must use the Weaver Qwen brain model');
check(speakerBody.includes('sole user-facing conversational speaker'), 'Public speaker boundary prompt is missing');
check(speakerBody.includes('Never identify as a coder'), 'Public speaker must reject coder identity');
check(!speakerBody.includes("quality reviewer"), 'Internal quality-reviewer identity must not be public');
check(!serialized.includes('No codebase evidence was supplied.'), 'Empty evidence must not poison conversational turns');
check(speakerTagCode.includes('speaker_boundary_applied: !!reviewed'), 'Speaker boundary must fail closed when the Weaver brain call fails');
check(speakerTagCode.includes('internal_draft_hidden: true'), 'Private specialist drafts must remain hidden');
check(writebackCode.includes(n8nContractVersion), 'Writeback must expose the versioned public contract');
check(writebackCode.includes("const PIPELINE = 'v6-parallel-cognition'"), 'Writeback must expose the v6 pipeline contract');
check(writebackCode.includes("pipeline_architecture: 'parallel-fanout-barrier'"), 'Writeback must expose the parallel topology');
check(writebackCode.includes("status: 'ok'") && writebackCode.includes('error: false'), 'Writeback must emit an explicit success discriminator');
check(writebackCode.includes("status: 'rejected'") && writebackCode.includes('error: true'), 'Writeback must emit an explicit rejection discriminator');
check(writebackCode.includes("speaker: 'weaver'"), 'Writeback must identify Weaver as the sole public speaker');
check(writebackCode.includes('speaker_boundary_applied: true'), 'Writeback must expose only a passed speaker boundary');
check(writebackCode.includes('internal_draft_hidden: true'), 'Writeback must attest that private drafts stayed hidden');
check(writebackCode.includes('manifested_response: reviewed'), 'Writeback must publish only the reviewed Weaver response');
check(!writebackCode.includes('...d'), 'Writeback must not spread internal pipeline state into public output');
check(!writebackCode.includes('d.collapsed_response ||'), 'Writeback must never fall back to a private specialist draft');
check(!writebackCode.includes('original_input:'), 'Writeback must not echo original input');
check(!writebackCode.includes('collapsed_response:'), 'Writeback must not duplicate private intermediate output');
for (const privateField of [
  'lora_error:', 'qwen3b_error:', 'dominant_lobe:', 'experts_activated:',
  'qwen3b_route:', 'synthesis_prompt:', 'codebase_context_chars:', 'qubit_layout:',
]) {
  check(!writebackCode.includes(privateField), `Writeback must not expose private field ${privateField.slice(0, -1)}`);
}

async function executeCodeNode(code, inputItems) {
  const input = {
    all: () => inputItems,
    first: () => inputItems[0],
  };
  const now = { toISO: () => '2026-07-13T12:00:00.000Z' };
  const forbiddenNamedLookup = () => {
    throw new Error('contract validation nodes may not use named-node lookups');
  };
  const callable = new AsyncFunction('$input', '$execution', '$workflow', '$now', '$', code);
  return callable(input, { id: 'validator-exec' }, {}, now, forbiddenNamedLookup);
}

const validContractRequest = {
  contract_version: n8nContractVersion,
  correlation_id: 'req-validator-1',
  deadline_ms: 115000,
  text: 'Hello Weaver',
  self_check: false,
  introspect: false,
  path_glob: '**/*',
  search_query: '',
  codebase_context: '',
  quantum_pathway: '',
  cognition_context: {
    awareness_confidence: 0.75,
    fabric_pressure: 0.1,
    immune_status: 'nominal',
    open_components: [],
  },
};
try {
  const [validSanitized] = await executeCodeNode(sanitizeCode, [{ json: validContractRequest }]);
  check(validSanitized?.json?.error !== true, 'Sanitize must accept the exact v1 request contract');
  check(validSanitized?.json?.correlation_id === validContractRequest.correlation_id, 'Sanitize must echo correlation IDs exactly');
  for (const mutation of [
    { ...validContractRequest, private_prompt: 'must-not-cross' },
    { ...validContractRequest, contract_version: 'legacy' },
    { ...validContractRequest, deadline_ms: 114999 },
    { ...validContractRequest, cognition_context: { ...validContractRequest.cognition_context, private_state: true } },
  ]) {
    const [rejected] = await executeCodeNode(sanitizeCode, [{ json: mutation }]);
    check(rejected?.json?.error_code === 'invalid-request', 'Sanitize must reject malformed or expanded contracts');
    check(!JSON.stringify(rejected).includes('must-not-cross'), 'Sanitize rejection must not echo untrusted fields');
  }

  const privateSentinel = 'PRIVATE-DRAFT-MUST-NOT-CROSS';
  const validInternalState = {
    ...validSanitized.json,
    self_reflection: 'Weaver public manifestation',
    reflection_applied: true,
    speaker_boundary_applied: true,
    speaker_model: 'qwen.qwen3-235b-a22b-2507',
    internal_draft_hidden: true,
    expert_parallel: true,
    expert_count: 5,
    experts_completed: 5,
    expert_errors: 0,
    expert_fanout_elapsed_ms: 102500,
    collapsed_response: privateSentinel,
    synthesis_prompt: privateSentinel,
    expert_drafts: [privateSentinel],
    lora_error: privateSentinel,
    qwen3b_error: privateSentinel,
    qwen3b_route: privateSentinel,
  };
  const [publicSuccess] = await executeCodeNode(writebackCode, [{ json: validInternalState }]);
  const successFields = [
    'contract_version', 'status', 'error', 'correlation_id', 'manifested_response',
    'speaker', 'speaker_boundary_applied', 'speaker_model', 'internal_draft_hidden',
    'reflection_applied', 'soul_voice_active', 'codebase_grounded', 'expert_parallel',
    'expert_count', 'experts_completed', 'expert_errors', 'expert_fanout_elapsed_ms',
    'execution_id', 'timestamp', 'pipeline_architecture', 'pipeline_version',
  ].sort();
  check(
    JSON.stringify(Object.keys(publicSuccess?.json || {}).sort()) === JSON.stringify(successFields),
    'Writeback success must contain exactly the public v1 fields',
  );
  check(publicSuccess?.json?.manifested_response === validInternalState.self_reflection, 'Writeback must publish the reviewed manifestation verbatim');
  check(!JSON.stringify(publicSuccess).includes(privateSentinel), 'Writeback must contain no private draft, route, prompt, or raw error');

  const [boundaryRejection] = await executeCodeNode(writebackCode, [{
    json: { ...validInternalState, speaker_boundary_applied: false },
  }]);
  const rejectionFields = [
    'contract_version', 'status', 'error', 'error_code', 'correlation_id',
    'execution_id', 'timestamp', 'pipeline_version',
  ].sort();
  check(
    JSON.stringify(Object.keys(boundaryRejection?.json || {}).sort()) === JSON.stringify(rejectionFields),
    'Writeback rejection must contain exactly the public rejection fields',
  );
  check(boundaryRejection?.json?.error_code === 'speaker-boundary-failed', 'Writeback must fail closed at the public speaker boundary');
  check(!Object.hasOwn(boundaryRejection?.json || {}, 'manifested_response'), 'Rejected output must never contain a manifestation');
} catch (error) {
  errors.push(`versioned n8n contract execution checks failed: ${error.message}`);
}

const settings = workflow.settings || {};
check(settings.executionOrder === 'v1', 'workflow executionOrder must be v1');
check(settings.saveExecutionProgress === false, 'workflow must not persist intermediate execution progress');
check(settings.saveManualExecutions === false, 'workflow must not persist manual executions by default');
const workflowTimeoutMs = Number(settings.executionTimeout) * 1000;
check(Number.isFinite(workflowTimeoutMs) && workflowTimeoutMs >= 30_000 && workflowTimeoutMs <= 120_000, 'workflow executionTimeout must be 30..120 seconds');

function attemptBudget(name) {
  const node = nodes.get(name);
  const timeout = Number(node?.parameters?.options?.timeout || 0);
  const tries = Number(node?.maxTries || 1);
  const wait = Number(node?.waitBetweenTries || 0);
  return timeout * tries + wait * Math.max(0, tries - 1);
}
const expertBudget = Math.max(...[...expectedExperts].map(attemptBudget));
const internetBudget = attemptBudget('Internet Context Bridge');
const reviewBudget = attemptBudget('7. Self-Reflect');
const localBudget = Math.max(attemptBudget('8. LoRA Voice'), attemptBudget('8b. Qwen3B'));
const criticalPathBudgetMs = expertBudget + internetBudget + reviewBudget + localBudget;
check(criticalPathBudgetMs <= workflowTimeoutMs, 'worst-case parallel critical path exceeds workflow executionTimeout');
check(criticalPathBudgetMs <= 115_000, 'worst-case n8n critical path exceeds the brain webhook budget');
warn(workflowTimeoutMs - criticalPathBudgetMs >= 8_000, 'workflow deadline headroom is below 8 seconds');

const fingerprint = createHash('sha256').update(raw).digest('hex');
const result = {
  valid: errors.length === 0,
  technology: 'weaver-n8n-contract-validator',
  version: 2,
  workflow: workflowPath,
  workflow_name: workflow.name,
  workflow_id: workflow.id || null,
  fingerprint_sha256: fingerprint,
  nodes: nodes.size,
  edges: edgeCount,
  code_nodes: codeNodeCount,
  expressions: expressionCount,
  reachable_nodes: reachable.size,
  terminal_nodes: terminals,
  critical_path_budget_ms: criticalPathBudgetMs,
  workflow_timeout_ms: workflowTimeoutMs,
  checks: checks.length,
  errors,
  warnings,
};

if (jsonOutput || errors.length) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} else {
  process.stdout.write(
    `VALID n8n workflow: ${nodes.size} nodes, ${edgeCount} edges, `
    + `${codeNodeCount} Code nodes, ${expressionCount} expressions\n`
    + `critical path budget: ${criticalPathBudgetMs} ms / ${workflowTimeoutMs} ms\n`
    + `sha256: ${fingerprint}\n`,
  );
  if (warnings.length) process.stdout.write(`warnings: ${warnings.join('; ')}\n`);
}

process.exit(errors.length ? 1 : 0);
