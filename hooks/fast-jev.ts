import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
} from 'claude-code';

import { compact, reductionBound, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CallAction,
  CallDecision,
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  if (options['keepUnscored'] === true) config.keepUnscored = true;
  return config;
}

export type Verdict =
  | { kind: 'scored' }
  | { kind: 'nothing_to_prune'; bound: number }
  | { kind: 'capacity'; estimatedPercent: number };

/**
 * What the compaction outcome means. Reduction below `minReductionRatio` is
 * not a failure when the candidates could not have freed that much anyway
 * (`nothing_to_prune`); it is only a problem when the window would still be
 * over the compaction threshold afterwards (`capacity`), which is when the
 * built-in summary is worth its cost.
 */
export function verdict(
  result: CompactResult,
  config: Pick<HookConfig, 'minReductionRatio' | 'compactAtPercent'>,
  windowPercent: number | undefined,
): Verdict {
  const reduction = reductionRatio(result);
  if (reduction >= config.minReductionRatio) return { kind: 'scored' };
  const bound = reductionBound(result);
  if (windowPercent !== undefined) {
    const estimatedPercent = windowPercent * (1 - reduction);
    if (estimatedPercent >= config.compactAtPercent) return { kind: 'capacity', estimatedPercent };
  }
  if (bound < config.minReductionRatio) return { kind: 'nothing_to_prune', bound };
  return { kind: 'scored' };
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.result !== undefined) summary.result = tool.result;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
    ...(result.result !== undefined ? { result: result.result } : {}),
  };
}

const PORTABLE_TOOL_ID = /^[a-zA-Z0-9_-]+$/;

function toolIds(message: Message): string[] {
  return [
    ...message.toolUses.map((tool) => tool.tool_use_id),
    ...(message.toolResults ?? []).map((result) => result.tool_use_id),
  ];
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }

  const owners = new Map<string, Set<Message>>();
  const reservedIds = new Set<string>();
  for (const message of output) {
    for (const id of toolIds(message)) {
      reservedIds.add(id);
      const ownedBy = owners.get(id) ?? new Set<Message>();
      ownedBy.add(message);
      owners.set(id, ownedBy);
    }
  }

  const remappedIds = new Map<string, string>();
  let nextId = 1;
  for (const [id, ownedBy] of owners) {
    if (PORTABLE_TOOL_ID.test(id)) continue;
    const rebuiltOwners = [...ownedBy].filter((message) => !messages.has(message));
    if (rebuiltOwners.length === 0) continue;
    // Rebuilding an engine-owned counterpart can discard hidden blocks that SessionMessage does not expose.
    if (rebuiltOwners.length !== ownedBy.size) {
      throw new Error('cannot safely remap non-portable tool id across an engine handle');
    }
    let replacement = '';
    do {
      replacement = `fjc_${nextId++}`;
    } while (reservedIds.has(replacement));
    reservedIds.add(replacement);
    remappedIds.set(id, replacement);
  }

  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => {
        const summary = uses.get(tool) ?? toolUseSummary(tool);
        const tool_use_id = remappedIds.get(summary.tool_use_id);
        return tool_use_id ? { ...summary, tool_use_id } : summary;
      }),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map((result) => {
        const summary = results.get(result) ?? toolResultSummary(result);
        const tool_use_id = remappedIds.get(summary.tool_use_id);
        return tool_use_id ? { ...summary, tool_use_id } : summary;
      });
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

/** Scored decisions per tool, most dropped calls first; pinned calls were never asked about. */
function byTool(result: CompactResult): { tool: string; decisions: CallDecision[]; dropped: number }[] {
  const groups = new Map<string, CallDecision[]>();
  for (const d of result.decisions) {
    if (d.reason !== 'pinned') groups.set(d.tool, [...(groups.get(d.tool) ?? []), d]);
  }
  return [...groups]
    .map(([tool, decisions]) => ({ tool, decisions, dropped: count(decisions, 'drop_call') }))
    .sort((a, b) => b.dropped - a.dropped || a.tool.localeCompare(b.tool));
}

function count(decisions: readonly CallDecision[], action: CallAction): number {
  return decisions.filter((d) => d.action === action).length;
}

// A transcript row is cut at the terminal's width budget, and past five tools
// the breakdown stops being readable anyway; `/fast-jev` lists every tool.
const SUMMARY_TOOLS = 5;

/** The outcome as one transcript row: messages, reduction, dropped calls per tool, the rest of the counts. */
export function summarize(result: CompactResult): string {
  const { stats } = result;
  const dropped = byTool(result).filter((group) => group.dropped > 0);
  const tools = dropped.slice(0, SUMMARY_TOOLS).map((group) => `${group.tool} ${group.dropped}`);
  if (dropped.length > SUMMARY_TOOLS) tools.push(`+${dropped.length - SUMMARY_TOOLS} tools`);
  const tokens =
    stats.stateTokens >= 1000 ? `${(stats.stateTokens / 1000).toFixed(1)}k` : `${stats.stateTokens}`;
  return [
    `${stats.messagesBefore}→${stats.messagesAfter} msgs`,
    `chars -${percent(reductionRatio(result))}`,
    stats.callsDropped > 0 ? `dropped ${stats.callsDropped} (${tools.join(', ')})` : '',
    stats.resultsDropped > 0 ? `truncated ${stats.resultsDropped}` : '',
    stats.kept > 0 ? `kept ${stats.kept}` : '',
    stats.pinned > 0 ? `pinned ${stats.pinned}` : '',
    stats.unscored > 0 ? `unscored ${stats.unscored}` : '',
    `${stats.requests} req`,
    `state ${tokens} tok`,
  ]
    .filter(Boolean)
    .join(' · ');
}

/** min · median · max, two decimals: where Jev's probabilities for one tool sat. */
function spread(values: readonly number[]): string {
  const sorted = [...values].sort((a, b) => a - b);
  const median = (sorted[(sorted.length - 1) >> 1]! + sorted[sorted.length >> 1]!) / 2;
  return [sorted[0]!, median, sorted[sorted.length - 1]!].map((v) => v.toFixed(2)).join(' · ');
}

/** `/fast-jev`'s table: per tool, what happened to its calls and the probabilities behind it. */
export function decisionTable(result: CompactResult): string {
  const groups = byTool(result);
  if (groups.length === 0) return 'No tool calls were candidates.';
  return [
    '| tool | dropped | truncated | kept | p(call) min · med · max | p(result) min · med · max |',
    '|---|--:|--:|--:|---|---|',
    ...groups.map(
      ({ tool, decisions, dropped }) =>
        `| ${tool} | ${dropped} | ${count(decisions, 'drop_result')} | ${count(decisions, 'keep')} | ${spread(
          decisions.map((d) => d.keepCall),
        )} | ${spread(decisions.map((d) => d.keepResult))} |`,
    ),
  ].join('\n');
}

/** HH:mm in the host's local time zone. */
function clockTime(ms: number): string {
  const at = new Date(ms);
  return `${String(at.getHours()).padStart(2, '0')}:${String(at.getMinutes()).padStart(2, '0')}`;
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/**
 * The context window's fill before compaction, or undefined where the host does
 * not report it. Claude's hook loader requires a direct `$.session.usage()` call.
 */
async function contextPercent($: {
  session: { usage: () => Promise<{ context: { percent?: number } }> };
}): Promise<number | undefined> {
  const usage = await $.session.usage();
  const value = usage.context.percent;
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

type LastCompaction = { time: string; line: string; result?: CompactResult };

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  // True only while `/compact-jev` waits on its own compaction, so `/compact`,
  // the engine's auto-compaction and other plugins keep the built-in summary.
  let requested = false;
  // What `/fast-jev` shows. A module reload runs register again, so it starts empty.
  let last: LastCompaction | undefined;

  on('session.start', async ($, event, next) => {
    await $.command.register({
      name: 'fast-jev',
      description: 'Show the last fast-jev compaction per tool',
    });
    await $.command.register({
      name: 'compact-jev',
      description: 'Compact by dropping tool calls and results Jev judges stale',
    });
    return next(event);
  });

  on('command.run', { command: 'compact-jev' }, async ($) => {
    // A `session.compact` this plugin raises itself (`$.session.compact()`) skips
    // this plugin's own hook as re-entry, so run the built-in `/compact` and claim
    // the compaction core raises. `$.command.run` rejects inside the dispatch that
    // holds the turn; a timer dispatch runs once the command has answered.
    $.clock.after(0, async () => {
      requested = true;
      try {
        await $.command.run({ command: 'compact' });
      } catch (error) {
        $.ui.log(`compaction failed: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        requested = false;
      }
    });
    return { text: 'Compacting through Jev…' };
  });

  on('command.run', { command: 'fast-jev' }, async () => ({
    text: last
      ? [
          `**${last.time}** · ${last.line}`,
          ...(last.result
            ? [`state stage: ${last.result.stats.stateStage}`, decisionTable(last.result)]
            : []),
        ].join('\n\n')
      : 'No compaction since the plugin loaded.',
  }));

  // `/compact-jev` runs through the built-in `/compact`, the `manual` trigger;
  // the matcher keeps a `precompute` that lands meanwhile off Jev.
  on('session.compact', { trigger: 'manual' }, async ($, event, next) => {
    if (!requested) return next(event);
    // One transcript row per compaction, and the status line keeps the latest
    // outcome under the prompt until the next one replaces it.
    const report = async (line: string, status: string, result?: CompactResult) => {
      const time = clockTime(await $.clock.now());
      last = { time, line, ...(result ? { result } : {}) };
      $.ui.log(line);
      // The engine already prefixes a plugin's status line with its name.
      $.ui.status(`${time} · ${status}`);
    };
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      });
      const outcome = verdict(result, config, await contextPercent($));
      if (outcome.kind === 'capacity') {
        const window = `window would stay at ~${Math.round(outcome.estimatedPercent)}%`;
        await report(
          `fallback to built-in summary: ${window} · ${summarize(result)}`,
          `built-in summary: ${window}`,
          result,
        );
        return next(event);
      }
      const note =
        outcome.kind === 'nothing_to_prune'
          ? ` · candidates were ${percent(outcome.bound)} of the history, the rest is text`
          : '';
      const { stats } = result;
      await report(
        `${summarize(result)}${note}`,
        `${stats.messagesBefore}→${stats.messagesAfter} msgs · -${percent(reductionRatio(result))} · ${stats.callsDropped} dropped`,
        result,
      );
      return { messages };
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      await report(`fallback to built-in summary: ${reason}`, `built-in summary: ${reason}`);
      return next(event);
    }
  });
};

export { resolveOptions };
