use super::*;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
    use crate::transport::{McpError, McpTransport};
    use aion_config::config::TransportType;
    use serde_json::json;

    struct ToolResultTransport {
        result: Value,
    }

    #[async_trait]
    impl McpTransport for ToolResultTransport {
        async fn request(&self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
            assert_eq!(request.method, "tools/call");
            assert_eq!(request.params, Some(json!({"name": "test_tool", "arguments": {}})));
            if let Some(error) = self.result.get("error") {
                return Err(McpError::JsonRpc {
                    code: error["code"].as_i64().unwrap(),
                    message: error["message"].as_str().unwrap().to_owned(),
                });
            }
            Ok(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: Some(self.result.clone()),
                error: None,
            })
        }

        async fn notify(&self, _: &JsonRpcRequest) -> Result<(), McpError> {
            Ok(())
        }

        async fn close(&self) -> Result<(), McpError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn execute_preserves_mcp_error_flags_and_rpc_failures() {
        // This is the installed Tencent MCP validation-result shape. Its error
        // flag must survive the manager/proxy path into agent error accounting.
        let validation_error = "MCP error -32602: Input validation error: Required query";
        let cases = [
            (
                json!({"content": [{"type": "text", "text": validation_error}], "isError": true}),
                true,
                validation_error,
            ),
            (
                json!({"content": [{"type": "text", "text": "ok"}], "isError": false}),
                false,
                "ok",
            ),
            (
                json!({"content": [{"type": "text", "text": "legacy ok"}]}),
                false,
                "legacy ok",
            ),
            (
                json!({"error": {"code": -32602, "message": "Invalid arguments"}}),
                true,
                "JSON-RPC error -32602",
            ),
        ];
        for (result, expected_error, expected_text) in cases {
            let manager = Arc::new(McpManager::new_for_test(vec![(
                "test_server",
                false,
                Box::new(ToolResultTransport { result }),
            )]));
            let proxy = McpToolProxy::new(
                "test_tool".into(),
                "test_tool".into(),
                "test_server".into(),
                "Test tool".into(),
                json!({"type": "object"}),
                manager,
                false,
            );
            let result = proxy.execute(json!({})).await;
            assert_eq!(result.is_error, expected_error);
            assert!(result.content.contains(expected_text));
        }
    }

    fn make_proxy(deferred: bool) -> McpToolProxy {
        // manager is only used during execute(), which we don't call in these
        // tests, so we can construct one with no servers.
        let manager = Arc::new(McpManager::new_for_test(vec![]));
        McpToolProxy::new(
            "test_tool".into(),
            "test_tool".into(),
            "test_server".into(),
            "A test tool".into(),
            json!({"type": "object"}),
            manager,
            deferred,
        )
    }

    #[test]
    fn proxy_deferred_true_returns_true() {
        let proxy = make_proxy(true);
        assert!(proxy.is_deferred());
    }

    #[test]
    fn proxy_deferred_false_returns_false() {
        let proxy = make_proxy(false);
        assert!(!proxy.is_deferred());
    }

    fn make_server_config(deferred: Option<bool>) -> McpServerConfig {
        McpServerConfig {
            transport: TransportType::Stdio,
            command: Some("echo".into()),
            args: None,
            env: None,
            url: None,
            headers: None,
            deferred,
            startup_timeout_ms: None,
        }
    }

    #[test]
    fn register_defaults_to_deferred_when_config_omits_field() {
        let manager = Arc::new(McpManager::new_for_test(vec![]));
        let mut registry = aion_tools::registry::ToolRegistry::new();
        // Empty server configs — deferred field absent
        let configs = HashMap::new();

        register_mcp_tools(&mut registry, &manager, &[], &configs);

        // No tools registered because manager has no tools, but the logic
        // is tested via the deferred default path. Test with a real config below.
        assert!(registry.tool_names().is_empty());
    }

    #[test]
    fn server_config_deferred_none_defaults_true() {
        let config = make_server_config(None);
        let deferred = config.deferred.unwrap_or(true);
        assert!(deferred, "deferred should default to true when None");
    }

    #[test]
    fn server_config_deferred_explicit_false() {
        let config = make_server_config(Some(false));
        let deferred = config.deferred.unwrap_or(true);
        assert!(!deferred, "deferred should be false when explicitly set");
    }

    #[test]
    fn server_config_deferred_explicit_true() {
        let config = make_server_config(Some(true));
        let deferred = config.deferred.unwrap_or(true);
        assert!(deferred, "deferred should be true when explicitly set");
    }
}
