use aion_tools::{ToolDocumentation, truncate_utf8};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use aion_types::tool::ToolResult;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::json;

use super::config::{McpServerConfig, TransportType};
use super::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, InitializeResult, JsonRpcRequest, McpResource, McpToolDef,
    McpToolResult, ResourcesListResult, ResourcesReadResult, ToolsListResult,
};
use super::transport::sse::SseTransport;
use super::transport::stdio::StdioTransport;
use super::transport::streamable_http::StreamableHttpTransport;
use super::transport::{McpError, McpTransport};

const DEFAULT_STARTUP_TIMEOUT_MS: u64 = 30_000;
const MAX_TOOL_PAGES: usize = 64;
const MAX_TOOLS: usize = 4096;
const CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
const CATALOG_REFRESH_TIMEOUT: Duration = Duration::from_secs(3);

/// A connected MCP server with its discovered tools and capabilities
struct McpServer {
    #[allow(dead_code)]
    name: String,
    transport: Box<dyn McpTransport>,
    tools: RwLock<Vec<McpToolDef>>,
    instructions: Option<String>,
    /// Whether the server declared resources capability in its initialize response
    supports_resources: bool,
    supports_tools: bool,
}

/// Manages connections to multiple MCP servers
pub struct McpManager {
    servers: HashMap<String, McpServer>,
    /// Monotonically increasing request ID counter for all JSON-RPC calls
    next_id: AtomicU64,
    last_refresh: Mutex<Instant>,
}

impl McpManager {
    /// Connect to all configured MCP servers
    pub async fn connect_all(configs: &HashMap<String, McpServerConfig>) -> Result<Self, McpError> {
        Self::connect_all_with_connector(configs, |name, config| async move {
            Self::connect_server(&name, &config).await
        })
        .await
    }

    async fn connect_all_with_connector<F, Fut>(
        configs: &HashMap<String, McpServerConfig>,
        connector: F,
    ) -> Result<Self, McpError>
    where
        F: Fn(String, McpServerConfig) -> Fut,
        Fut: Future<Output = Result<McpServer, McpError>>,
    {
        let mut servers = HashMap::new();
        let mut pending = FuturesUnordered::new();

        for (name, config) in configs {
            let name = name.clone();
            let config = config.clone();
            let connect = connector(name.clone(), config.clone());
            pending.push(async move {
                let result = Self::with_startup_timeout(&name, &config, connect).await;
                (name, result)
            });
        }

        while let Some((name, result)) = pending.next().await {
            match result {
                Ok(server) => {
                    tracing::info!(target: "aion_mcp", server = %name, tools = server.tools.read().unwrap_or_else(|error| error.into_inner()).len(), resources = server.supports_resources, "mcp server connected");
                    servers.insert(name, server);
                }
                Err(e) => {
                    // Non-fatal: continue with other servers
                    tracing::warn!(target: "aion_mcp", server = %name, error = %e, "mcp server connection failed");
                }
            }
        }

        Ok(Self {
            servers,
            next_id: AtomicU64::new(10),
            last_refresh: Mutex::new(Instant::now()),
        })
    }

    fn startup_timeout(config: &McpServerConfig) -> Duration {
        Duration::from_millis(config.startup_timeout_ms.unwrap_or(DEFAULT_STARTUP_TIMEOUT_MS))
    }

    async fn with_startup_timeout<Fut>(
        name: &str,
        config: &McpServerConfig,
        connect: Fut,
    ) -> Result<McpServer, McpError>
    where
        Fut: Future<Output = Result<McpServer, McpError>>,
    {
        let timeout = Self::startup_timeout(config);
        match tokio::time::timeout(timeout, connect).await {
            Ok(result) => result,
            Err(_) => Err(McpError::Transport(format!(
                "MCP server '{name}' startup timed out after {}ms; set startup_timeout_ms to increase it",
                timeout.as_millis()
            ))),
        }
    }

    /// Connect a single additional MCP server after initial setup.
    /// Returns the list of tool names exposed by the server.
    pub async fn connect_one(&mut self, name: String, config: &McpServerConfig) -> Result<Vec<String>, McpError> {
        let server = Self::with_startup_timeout(&name, config, Self::connect_server(&name, config)).await?;
        let tool_names: Vec<String> = server
            .tools
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|t| t.name.clone())
            .collect();
        tracing::info!(target: "aion_mcp", server = %name, tools = server.tools.read().unwrap_or_else(|error| error.into_inner()).len(), resources = server.supports_resources, "mcp server connected");
        self.servers.insert(name, server);
        Ok(tool_names)
    }

    /// Connect to a single MCP server: create transport, initialize, discover tools
    async fn connect_server(name: &str, config: &McpServerConfig) -> Result<McpServer, McpError> {
        let empty_map = HashMap::new();

        // 1. Create transport
        let transport: Box<dyn McpTransport> = match config.transport {
            TransportType::Stdio => {
                let command = config
                    .command
                    .as_deref()
                    .ok_or_else(|| McpError::InitFailed("stdio transport requires 'command'".into()))?;
                let args = config.args.as_deref().unwrap_or(&[]);
                let env = config.env.as_ref().unwrap_or(&empty_map);
                Box::new(StdioTransport::spawn(command, args, env).await?)
            }
            TransportType::Sse => {
                let url = config
                    .url
                    .as_deref()
                    .ok_or_else(|| McpError::InitFailed("SSE transport requires 'url'".into()))?;
                let headers = config.headers.as_ref().unwrap_or(&empty_map);
                Box::new(SseTransport::connect(url, headers).await?)
            }
            TransportType::StreamableHttp => {
                let url = config
                    .url
                    .as_deref()
                    .ok_or_else(|| McpError::InitFailed("streamable-http transport requires 'url'".into()))?;
                let headers = config.headers.as_ref().unwrap_or(&empty_map);
                Box::new(StreamableHttpTransport::connect(url, headers).await?)
            }
        };

        Self::initialize_transport(name, transport).await
    }

    async fn initialize_transport(name: &str, transport: Box<dyn McpTransport>) -> Result<McpServer, McpError> {
        // 2. Initialize handshake
        let init_params = InitializeParams {
            protocol_version: "2025-03-26".to_string(),
            capabilities: ClientCapabilities { tools: Some(json!({})) },
            client_info: ClientInfo {
                name: "aionrs".to_string(),
                version: "0.3.0".to_string(),
            },
        };

        let init_req = JsonRpcRequest::new(
            1,
            "initialize",
            Some(
                serde_json::to_value(&init_params)
                    .map_err(|e| McpError::InitFailed(format!("Failed to serialize init params: {}", e)))?,
            ),
        );

        let init_response = transport.request(&init_req).await?;
        let init_result: InitializeResult = serde_json::from_value(
            init_response
                .result
                .ok_or_else(|| McpError::InitFailed("No result in initialize response".into()))?,
        )
        .map_err(|e| McpError::InitFailed(format!("Failed to parse init result: {}", e)))?;

        // Check whether server declared resources capability
        let supports_resources = init_result
            .capabilities
            .get("resources")
            .map(|v| !v.is_null())
            .unwrap_or(false);

        // 3. Send initialized notification
        let initialized_notification = JsonRpcRequest::notification("notifications/initialized", None);
        transport.notify(&initialized_notification).await?;

        // 4. Fully validate the catalog before publishing any tools.
        let supports_tools = init_result
            .capabilities
            .get("tools")
            .is_some_and(|value| !value.is_null());
        let tools = if supports_tools {
            Self::list_tools(transport.as_ref(), &AtomicU64::new(2)).await?
        } else {
            Vec::new()
        };
        let instructions = init_result
            .instructions
            .filter(|text| !text.trim().is_empty())
            .map(|text| truncate_utf8(&text, 16_000).to_string());
        Ok(McpServer {
            name: name.to_string(),
            transport,
            tools: RwLock::new(tools),
            instructions,
            supports_resources,
            supports_tools,
        })
    }

    async fn list_tools(transport: &dyn McpTransport, next_id: &AtomicU64) -> Result<Vec<McpToolDef>, McpError> {
        let mut tools = Vec::new();
        let mut names = HashSet::new();
        let mut cursors = HashSet::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_TOOL_PAGES {
            let params = cursor.as_ref().map(|cursor| json!({"cursor": cursor}));
            let response = transport
                .request(&JsonRpcRequest::new(
                    next_id.fetch_add(1, Ordering::Relaxed),
                    "tools/list",
                    params,
                ))
                .await?;
            if let Some(error) = response.error {
                return Err(McpError::JsonRpc {
                    code: error.code,
                    message: error.message,
                });
            }
            let page: ToolsListResult = serde_json::from_value(
                response
                    .result
                    .ok_or_else(|| McpError::InitFailed("No result in tools/list response".into()))?,
            )
            .map_err(|error| McpError::InitFailed(format!("Invalid tools/list response: {error}")))?;
            for tool in page.tools {
                if tool.name.trim().is_empty() || !names.insert(tool.name.clone()) {
                    return Err(McpError::InitFailed(
                        "Empty or duplicate tool name in tools/list".into(),
                    ));
                }
                tools.push(tool);
                if tools.len() > MAX_TOOLS {
                    return Err(McpError::InitFailed("MCP tool catalog exceeds supported size".into()));
                }
            }
            match page.next_cursor {
                None => {
                    tools.sort_by(|a, b| a.name.cmp(&b.name));
                    return Ok(tools);
                }
                Some(next) if cursors.insert(next.clone()) => cursor = Some(next),
                Some(_) => return Err(McpError::InitFailed("Repeated tools/list cursor".into())),
            }
        }
        Err(McpError::InitFailed("MCP tools/list pagination limit exceeded".into()))
    }

    /// Refresh at an idle model-turn boundary. Never changes a catalog during tool execution.
    /// Notification flags avoid polling; the TTL also covers transports without an idle event stream.
    /// Failed/partial refreshes retain the last complete snapshot and retry after the TTL.
    pub async fn refresh_tools_if_due(&self) -> bool {
        let mut notified = false;
        for server in self.servers.values() {
            notified |= server.transport.take_tools_changed();
        }
        {
            let mut last = self.last_refresh.lock().unwrap_or_else(|error| error.into_inner());
            if !notified && last.elapsed() < CATALOG_REFRESH_INTERVAL {
                return false;
            }
            *last = Instant::now();
        }
        let mut pending = FuturesUnordered::new();
        for (name, server) in &self.servers {
            if !server.supports_tools {
                continue;
            }
            pending.push(async move {
                (
                    name,
                    server,
                    tokio::time::timeout(
                        CATALOG_REFRESH_TIMEOUT,
                        Self::list_tools(server.transport.as_ref(), &self.next_id),
                    )
                    .await,
                )
            });
        }
        let mut changed = false;
        while let Some((name, server, result)) = pending.next().await {
            match result {
                Ok(Ok(tools)) => {
                    let mut current = server.tools.write().unwrap_or_else(|error| error.into_inner());
                    if *current != tools {
                        tracing::info!(target: "aion_mcp", server = %name, tools = tools.len(), "MCP catalog refreshed");
                        *current = tools;
                        changed = true;
                    }
                }
                _ => {
                    tracing::warn!(target: "aion_mcp", server = %name, "MCP catalog refresh failed; retaining previous complete catalog")
                }
            }
        }
        changed
    }

    /// Snapshot discovered tools. The owned snapshot remains valid during catalog refresh.
    pub fn all_tools(&self) -> Vec<(String, McpToolDef)> {
        let mut result = Vec::new();
        for (name, server) in &self.servers {
            for tool in server.tools.read().unwrap_or_else(|error| error.into_inner()).iter() {
                result.push((name.clone(), tool.clone()));
            }
        }
        result.sort_by(|a, b| (&a.0, &a.1.name).cmp(&(&b.0, &b.1.name)));
        result
    }

    /// Documentation belongs to this initialized connection and is not a system policy.
    pub fn server_documentation(&self, server_name: &str) -> Option<ToolDocumentation> {
        let content = self.servers.get(server_name)?.instructions.clone()?;
        Some(ToolDocumentation {
            source: server_name.to_string(),
            content,
        })
    }

    pub fn has_tool_name(&self, name: &str) -> bool {
        self.tool_name_count(name) > 0
    }

    pub fn tool_name_count(&self, name: &str) -> usize {
        self.servers
            .values()
            .filter(|server| {
                server
                    .tools
                    .read()
                    .unwrap_or_else(|error| error.into_inner())
                    .iter()
                    .any(|tool| tool.name == name)
            })
            .count()
    }

    /// Execute a tool on a specific server
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult, McpError> {
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        let request = JsonRpcRequest::new(
            self.next_id.fetch_add(1, Ordering::Relaxed),
            "tools/call",
            Some(json!({
                "name": tool_name,
                "arguments": arguments
            })),
        );

        let response = server.transport.request(&request).await?;

        let result_value = response
            .result
            .ok_or_else(|| McpError::Transport("No result in tool call response".into()))?;

        // Parse result and concatenate text content
        let tool_result: McpToolResult = serde_json::from_value(result_value)
            .map_err(|e| McpError::Transport(format!("Failed to parse tool result: {}", e)))?;

        let mut text_parts = Vec::new();
        for content in &tool_result.content {
            match content {
                super::protocol::McpContent::Text { text } => text_parts.push(text.clone()),
                super::protocol::McpContent::Image { mime_type, .. } => {
                    text_parts.push(format!("[image: {}]", mime_type));
                }
                super::protocol::McpContent::Resource { .. } => {
                    text_parts.push("[resource]".to_string());
                }
            }
        }

        if tool_result.is_error {
            tracing::warn!(target: "aion_mcp", server = server_name, tool = tool_name, "mcp tool returned an error result");
        }
        Ok(ToolResult {
            content: text_parts.join("\n"),
            is_error: tool_result.is_error,
        })
    }

    /// Get names of all connected servers.
    pub fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Check if a connected server declared the resources capability.
    pub fn server_supports_resources(&self, server_name: &str) -> bool {
        self.servers
            .get(server_name)
            .map(|s| s.supports_resources)
            .unwrap_or(false)
    }

    /// List all resources from a server.
    pub async fn list_resources(&self, server_name: &str) -> Result<Vec<McpResource>, McpError> {
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, "resources/list", None);
        let response = server.transport.request(&request).await?;

        let result_value = response
            .result
            .ok_or_else(|| McpError::Transport("No result in resources/list response".into()))?;

        let list_result: ResourcesListResult = serde_json::from_value(result_value)
            .map_err(|e| McpError::Transport(format!("Failed to parse resources/list: {}", e)))?;

        Ok(list_result.resources)
    }

    /// Read a single resource by URI from a server. Returns the text content.
    pub async fn read_resource(&self, server_name: &str, uri: &str) -> Result<String, McpError> {
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, "resources/read", Some(json!({ "uri": uri })));
        let response = server.transport.request(&request).await?;

        let result_value = response
            .result
            .ok_or_else(|| McpError::Transport("No result in resources/read response".into()))?;

        let read_result: ResourcesReadResult = serde_json::from_value(result_value)
            .map_err(|e| McpError::Transport(format!("Failed to parse resources/read: {}", e)))?;

        // Return the first text content found
        read_result
            .contents
            .into_iter()
            .find_map(|c| c.text)
            .ok_or_else(|| McpError::Transport(format!("No text content in resource '{}'", uri)))
    }

    /// Gracefully shutdown all servers
    pub async fn shutdown(&self) {
        for (name, server) in &self.servers {
            if let Err(e) = server.transport.close().await {
                tracing::warn!(target: "aion_mcp", server = %name, error = %e, "error closing mcp server");
            }
        }
    }

    /// Test-only constructor: build a manager from pre-configured servers.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn new_for_test(entries: Vec<(&str, bool, Box<dyn super::transport::McpTransport>)>) -> Self {
        let mut servers = HashMap::new();
        for (name, supports_resources, transport) in entries {
            servers.insert(
                name.to_string(),
                McpServer {
                    name: name.to_string(),
                    transport,
                    tools: RwLock::new(vec![]),
                    instructions: None,
                    supports_resources,
                    supports_tools: true,
                },
            );
        }
        Self {
            servers,
            next_id: AtomicU64::new(10),
            last_refresh: Mutex::new(Instant::now()),
        }
    }
}

#[cfg(test)]
#[path = "manager_test.rs"]
mod manager_test;
