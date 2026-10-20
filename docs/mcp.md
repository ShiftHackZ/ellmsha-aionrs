# MCP (Model Context Protocol) Integration

## Overview

MCP allows the agent to connect to external tool servers, extending beyond the 7 built-in tools to the entire MCP server ecosystem.

## Configuring MCP Servers

Declare MCP servers in the config file:

```toml
# Stdio transport: launch a local subprocess
[mcp.servers.filesystem]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/project"]

[mcp.servers.github]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_TOKEN = "ghp_xxx" }
startup_timeout_ms = 30000

# SSE transport: connect to a remote SSE server
[mcp.servers.database]
transport = "sse"
url = "http://localhost:3001/sse"

# Streamable HTTP transport: HTTP POST communication
[mcp.servers.remote-tools]
transport = "streamable-http"
url = "https://tools.example.com/mcp"
headers = { Authorization = "Bearer xxx" }
```

## Transport Types

| Transport | Description | Use Case |
|-----------|-------------|----------|
| `stdio` | Launch local subprocess, communicate via stdin/stdout | Local MCP servers (npx, uvx) |
| `sse` | GET for SSE event stream, POST for requests | Remote MCP servers |
| `streamable-http` | HTTP POST, supports SSE streaming responses | Remote MCP servers |

## Startup Timeout

Configured MCP servers are connected concurrently during startup. Each server
has a startup timeout covering transport connection, `initialize`, and
`tools/list`. The default is `30000` milliseconds.

```toml
[mcp.servers.slow-tools]
transport = "stdio"
command = "npx"
args = ["-y", "slow-mcp-server"]
startup_timeout_ms = 60000
```

Increase `startup_timeout_ms` for servers that need extra time for first-run
setup, package downloads, remote authentication, or slow network handshakes.

## Deferred Loading

MCP tools can be registered as "deferred" — their full schema is not sent to the model at startup, reducing initial token usage. The LLM discovers deferred tools via the `ToolSearch` tool when needed.

```toml
[mcp.servers.large-toolset]
transport = "stdio"
command = "npx"
args = ["-y", "my-mcp-server"]
deferred = true    # Don't load tool schemas at startup
```

| `deferred` | Behavior |
|------------|----------|
| `false` | Tool schemas included in provider requests at startup |
| `true` (default) | Tools registered but schemas loaded on-demand via ToolSearch |

Deferred tools stay in the authorized host-side search index, entirely outside
initial provider tool definitions. `ToolSearch` promotes only its selected page
of full schemas into subsequent requests. Discovery state belongs to one agent;
parents, children and skill forks cannot load schemas into each other's context.
If a restricted tool policy permits a tool but denies ToolSearch, its full schema
is advertised directly so that explicit permission does not become unusable.
The desktop host enables deferral for every injected MCP server.

Skills use a bounded name/description catalog; full bodies enter model context
through `Skill` only when selected. Ordinary native children now retain the Skill
tool and managed skill directories as well as inherited MCP connections.

## Tool Naming

- MCP tool names are used directly when there's no conflict
- On conflict with built-in or other MCP tools, names are auto-prefixed: `mcp__{server}__{tool}`

## Startup Flow

1. Connect to all configured MCP servers
2. Perform MCP protocol handshake (`initialize`) for each server
3. Discover available tools (`tools/list`)
4. Register tools in the tool registry — the agent uses them like built-in tools
5. Gracefully close all connections on exit

## Scoped Integration Documentation and Catalog Refresh

The native runtime retains optional `initialize.instructions` as server-provided
**tool documentation**, not system policy. `ToolSearch` returns documentation only
for matching, policy-allowed tools, once per integration in each result. Search by
a tool name, integration name, or documentation keyword. The default result page
contains 10 schemas (maximum 25); use `next_offset` to continue. Documentation is
bounded to 8 KB per integration and 16 KB per result, with truncation indicated.
Repeated searches do not append documentation to the system prompt. Servers that
omit instructions continue to work unchanged.

`tools/list` follows up to 64 pages / 4,096 tools, rejecting repeated cursors and
duplicate names rather than accepting a partial catalog. Between completed model
rounds, a tools-changed notification or a 60-second TTL triggers discovery. Servers
refresh concurrently with a three-second total budget per server; a failed refresh
retains its last complete catalog. Successful refreshes replace removed tools and
changed schemas in the existing registry and rebuild the filtered ToolSearch
snapshot. Changed or removed schemas lose their loaded state and require discovery again. Active tool calls are never interrupted for refresh.

The TTL covers Streamable HTTP without an idle notification stream and buffered
stdio notifications. There is no background polling or automatic server restart.
New connections obtain their own instructions; changing initialization instructions
requires reconnecting the server. External agent clients manage their own MCP
catalogs separately.
