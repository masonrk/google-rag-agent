import 'dotenv/config';
import { FunctionTool, LlmAgent, MCPToolset } from '@google/adk';
import { google } from 'googleapis';
import { z } from 'zod';

type McpServerConfig =
  | {
      name: string;
      type: 'http';
      url: string;
      headers?: Record<string, string>;
      toolPrefix?: string;
    }
  | {
      name: string;
      type: 'stdio';
      command: string;
      args?: string[];
      env?: Record<string, string>;
      toolPrefix?: string;
    };

const DEFAULT_MODEL = process.env.AGENT_MODEL ?? 'gemini-2.5-flash';
const DEFAULT_RAG_CORPUS =
  process.env.VERTEX_RAG_CORPUS ??
  'projects/gcloud-moosylvania/locations/us-east5/ragCorpora/4611686018427387904';
const USER_PROFILE = {
  name: process.env.USER_NAME ?? null,
  workEmail: process.env.USER_WORK_EMAIL ?? null,
};
const APP_USER_EMAIL_KEYS = ['app:user_email', 'user_email', 'user:email'];
const APP_USER_NAME_KEYS = ['app:user_name', 'user_name', 'user:name'];

function parseJsonEnv<T>(name: string, fallback: T): T {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }

  try {
    return JSON.parse(value) as T;
  } catch (error) {
    throw new Error(`Environment variable ${name} must be valid JSON: ${String(error)}`);
  }
}

function parseParentFromCorpus(ragCorpus: string): string {
  const match = ragCorpus.match(/^(projects\/[^/]+\/locations\/[^/]+)\/ragCorpora\/[^/]+$/);
  if (!match) {
    throw new Error(
      `Invalid Vertex RAG corpus name: ${ragCorpus}. Expected projects/{project}/locations/{location}/ragCorpora/{id}.`,
    );
  }

  return match[1];
}

function parseLocationFromCorpus(ragCorpus: string): string {
  const match = ragCorpus.match(/^projects\/[^/]+\/locations\/([^/]+)\/ragCorpora\/[^/]+$/);
  if (!match) {
    throw new Error(
      `Invalid Vertex RAG corpus name: ${ragCorpus}. Expected projects/{project}/locations/{location}/ragCorpora/{id}.`,
    );
  }

  return match[1];
}

function buildMcpToolsets(): MCPToolset[] {
  const servers = parseJsonEnv<McpServerConfig[]>('MCP_SERVERS_JSON', []);

  return servers.map((server) => {
    if (server.type === 'http') {
      return new MCPToolset(
        {
          type: 'StreamableHTTPConnectionParams',
          url: server.url,
          transportOptions: server.headers
            ? {
                requestInit: {
                  headers: server.headers,
                },
              }
            : undefined,
        },
        [],
        server.toolPrefix ?? server.name,
      );
    }

    return new MCPToolset(
      {
        type: 'StdioConnectionParams',
        serverParams: {
          command: server.command,
          args: server.args ?? [],
          env: server.env,
        },
      },
      [],
      server.toolPrefix ?? server.name,
    );
  });
}

function getFirstStringValue(
  source: { get: <T>(key: string, defaultValue?: T) => T | undefined },
  keys: string[],
): string | null {
  for (const key of keys) {
    const value = source.get<string>(key);
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }

  return null;
}

function resolveIdentityFromState(
  source: { get: <T>(key: string, defaultValue?: T) => T | undefined },
): { name: string | null; email: string | null; source: string } {
  const stateEmail = getFirstStringValue(source, APP_USER_EMAIL_KEYS);
  const stateName = getFirstStringValue(source, APP_USER_NAME_KEYS);

  if (stateEmail || stateName) {
    return {
      name: stateName,
      email: stateEmail,
      source: 'session_state',
    };
  }

  return {
    name: USER_PROFILE.name,
    email: USER_PROFILE.workEmail,
    source: USER_PROFILE.name || USER_PROFILE.workEmail ? 'env' : 'none',
  };
}

async function retrieveRagContexts(query: string, ragCorpus: string, topK: number) {
  const auth = new google.auth.GoogleAuth({
    scopes: ['https://www.googleapis.com/auth/cloud-platform'],
  });

  google.options({ auth });

  const location = parseLocationFromCorpus(ragCorpus);
  const aiPlatform = google.aiplatform({
    version: 'v1beta1',
    rootUrl: `https://${location}-aiplatform.googleapis.com/`,
  });
  const response = await aiPlatform.projects.locations.retrieveContexts({
    parent: parseParentFromCorpus(ragCorpus),
    requestBody: {
      query: {
        text: query,
      },
      vertexRagStore: {
        ragResources: [{ ragCorpus }],
        vectorDistanceThreshold: Number(process.env.VERTEX_RAG_DISTANCE_THRESHOLD ?? '0.7'),
      },
    },
  });

  const contexts = response.data.contexts?.contexts ?? [];
  return contexts.slice(0, topK);
}

function isEmailQuestion(text: string): boolean {
  const normalized = text.toLowerCase();
  return normalized.includes('my email') || normalized.includes('work email') || normalized.includes('email address');
}

function buildRagSummary(
  query: string,
  ragCorpus: string,
  contexts: Array<{ text?: string | null; sourceUri?: string | null; distance?: number | null }>,
): string {
  if (contexts.length === 0) {
    return `No matching context was returned from ${ragCorpus} for query: ${query}`;
  }

  return contexts
    .map((context, index) => {
      const source = context.sourceUri ?? 'unknown source';
      const distance = context.distance != null ? `distance=${context.distance}` : 'distance=unknown';
      const excerpt = (context.text ?? '').replace(/\s+/g, ' ').trim();
      return `${index + 1}. [${source}] ${distance} ${excerpt}`;
    })
    .join('\n');
}

const listConnectedApps = new FunctionTool({
  name: 'list_connected_apps',
  description: 'Lists the external applications currently connected to the agent through MCP.',
  execute: () => {
    const servers = parseJsonEnv<McpServerConfig[]>('MCP_SERVERS_JSON', []);

    return {
      connected_apps: servers.map((server) => ({
        name: server.name,
        type: server.type,
        tool_prefix: server.toolPrefix ?? server.name,
      })),
      total: servers.length,
    };
  },
});

const getUserProfile = new FunctionTool({
  name: 'get_user_profile',
  description:
    'Returns the current user identity for the request. Prefers app-supplied session state like app:user_email and app:user_name, then falls back to local environment values.',
  execute: (_input, context) => {
    const resolved = context ? resolveIdentityFromState(context.state) : {
      name: USER_PROFILE.name,
      email: USER_PROFILE.workEmail,
      source: USER_PROFILE.name || USER_PROFILE.workEmail ? 'env' : 'none',
    };

    return {
      name: resolved.name,
      work_email: resolved.email,
      has_name: Boolean(resolved.name),
      has_work_email: Boolean(resolved.email),
      identity_source: resolved.source,
      accepted_state_keys: {
        user_email: APP_USER_EMAIL_KEYS,
        user_name: APP_USER_NAME_KEYS,
      },
    };
  },
});

const searchRagCorpus = new FunctionTool({
  name: 'search_rag_corpus',
  description:
    'Searches the configured Vertex AI RAG corpus and returns the most relevant retrieved context chunks with source information.',
  parameters: z.object({
    query: z.string().min(1).describe('The question or search query to run against the RAG corpus.'),
    ragCorpus: z
      .string()
      .default(DEFAULT_RAG_CORPUS)
      .describe('Optional full Vertex RAG corpus resource name.'),
    topK: z.number().int().min(1).max(10).default(5).describe('Maximum number of chunks to return.'),
  }),
  execute: async ({ query, ragCorpus = DEFAULT_RAG_CORPUS, topK = 5 }) => {
    const contexts = await retrieveRagContexts(query, ragCorpus, topK);

    return {
      query,
      rag_corpus: ragCorpus,
      result_count: contexts.length,
      results: contexts.map((context) => ({
        source_uri: context.sourceUri ?? null,
        distance: context.distance ?? null,
        text: context.text ?? '',
      })),
      summary: buildRagSummary(query, ragCorpus, contexts),
    };
  },
});

const rootTools = [getUserProfile, listConnectedApps, searchRagCorpus, ...buildMcpToolsets()];

export const rootAgent = new LlmAgent({
  name: 'workspace_orchestrator_agent',
  model: DEFAULT_MODEL,
  description:
    'A general-purpose orchestration agent that can answer normal questions, query a Vertex AI RAG corpus, and work across multiple connected applications.',
  instruction: `
You are a general-purpose assistant for business and knowledge workflows.

Behavior rules:
- Use search_rag_corpus whenever the user is asking for company knowledge, documentation, internal reference material, or anything likely stored in the RAG corpus.
- When you answer from the RAG corpus, cite the returned source URIs in plain text.
- Use get_user_profile when the user asks about their identity, email, name, or request-specific profile details.
- For email questions, check get_user_profile first. Treat app-supplied session state as the canonical identity source for the current user.
- Use the connected application tools when the user asks to inspect or act on external systems.
- If no connected application is available for the request, say that clearly and suggest using list_connected_apps.
- You are not limited to the RAG corpus. For ordinary conversation or general reasoning, answer directly.
- Be concise, accurate, and action-oriented.
`.trim(),
  beforeModelCallback: async ({ request }) => {
    const message =
      request.contents
        ?.flatMap((content) => content.parts ?? [])
        .map((part) => ('text' in part ? part.text ?? '' : ''))
        .join(' ')
        .trim() ?? '';

    if (isEmailQuestion(message) && !USER_PROFILE.workEmail) {
      return {
        content: {
          role: 'model',
          parts: [
            {
              text:
                'I do not have a request-specific email yet. For an organization app, pass the current user email in session state using app:user_email. You can also use USER_WORK_EMAIL as a local fallback in development.',
            },
          ],
        },
      };
    }

    return undefined;
  },
  tools: rootTools,
});

export default rootAgent;
