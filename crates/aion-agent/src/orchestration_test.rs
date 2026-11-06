use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- truncate_display -----------------------------------------------------

    #[test]
    fn truncate_display_ascii_short_unchanged() {
        assert_eq!(truncate_display("hello", 10), "hello");
    }

    #[test]
    fn truncate_display_ascii_truncated() {
        let result = truncate_display("hello world", 5);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 20);
    }

    #[test]
    fn truncate_display_cjk_does_not_panic() {
        // 200 CJK chars: each is 3 bytes, so byte index 200 falls mid-character
        let cjk: String = "你好世界测试".chars().cycle().take(200).collect();
        let result = truncate_display(&cjk, 50);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn truncate_display_mixed_cjk_ascii_does_not_panic() {
        let mixed = "abc你好def世界ghi测试".repeat(20);
        let result = truncate_display(&mixed, 30);
        assert!(result.ends_with("..."));
    }

    // -- truncate_result ------------------------------------------------------

    #[test]
    fn truncate_result_short_unchanged() {
        let s = "short content";
        assert_eq!(truncate_result(s, 1000), s);
    }

    #[test]
    fn truncate_result_cjk_does_not_panic() {
        let cjk: String = "这是一段较长的中文内容用于测试截断功能".repeat(50);
        let result = truncate_result(&cjk, 100);
        assert!(result.contains("truncated"));
    }

    #[test]
    fn truncate_result_mixed_cjk_ascii_does_not_panic() {
        let mixed = "Hello你好World世界Test测试".repeat(100);
        let result = truncate_result(&mixed, 200);
        assert!(result.contains("truncated"));
    }

    // -- maybe_append_deferred_hint -------------------------------------------

    #[test]
    fn deferred_hint_appended_when_required_field_missing() {
        let schema = json!({
            "type": "object",
            "properties": { "tasks": { "type": "array" } },
            "required": ["tasks"]
        });
        let input = json!({});
        let result = maybe_append_deferred_hint("Missing or invalid 'tasks' array", schema, &input);
        assert!(result.contains("Missing or invalid 'tasks' array"));
        assert!(result.contains("ToolSearch"));
    }

    #[test]
    fn deferred_hint_not_appended_when_required_fields_present() {
        let schema = json!({
            "type": "object",
            "properties": { "tasks": { "type": "array" } },
            "required": ["tasks"]
        });
        let input = json!({"tasks": [{"name": "t1", "prompt": "do x"}]});
        let result = maybe_append_deferred_hint("Some runtime error", schema, &input);
        assert_eq!(result, "Some runtime error");
        assert!(!result.contains("ToolSearch"));
    }

    #[test]
    fn deferred_hint_not_appended_when_no_required_field() {
        let schema = json!({
            "type": "object",
            "properties": {}
        });
        let input = json!({});
        let result = maybe_append_deferred_hint("some error", schema, &input);
        assert_eq!(result, "some error");
    }

    #[test]
    fn deferred_hint_not_appended_when_required_is_empty() {
        let schema = json!({
            "type": "object",
            "properties": {},
            "required": []
        });
        let input = json!({});
        let result = maybe_append_deferred_hint("some error", schema, &input);
        assert_eq!(result, "some error");
    }

    #[test]
    fn deferred_hint_appended_for_partial_missing_fields() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "string" }
            },
            "required": ["a", "b"]
        });
        let input = json!({"a": "present"});
        let result = maybe_append_deferred_hint("validation failed", schema, &input);
        assert!(result.contains("ToolSearch"));
    }

    // -- execute_single integration tests (deferred tool hint) ----------------

    use aion_tools::Tool;
    use aion_tools::registry::ToolRegistry;

    struct MockDeferredTool {
        schema: serde_json::Value,
    }

    #[async_trait::async_trait]
    impl Tool for MockDeferredTool {
        fn name(&self) -> &str {
            "MockDeferred"
        }
        fn description(&self) -> &str {
            "A mock deferred tool for testing"
        }
        fn input_schema(&self) -> serde_json::Value {
            self.schema.clone()
        }
        fn is_concurrency_safe(&self, _input: &serde_json::Value) -> bool {
            true
        }
        fn is_deferred(&self) -> bool {
            true
        }
        async fn execute(&self, input: serde_json::Value) -> aion_types::tool::ToolResult {
            if input.get("tasks").is_none() {
                return aion_types::tool::ToolResult {
                    content: "Missing or invalid 'tasks' array".to_string(),
                    is_error: true,
                };
            }
            aion_types::tool::ToolResult {
                content: "ok".to_string(),
                is_error: false,
            }
        }
        fn category(&self) -> aion_protocol::events::ToolCategory {
            aion_protocol::events::ToolCategory::Exec
        }
    }

    struct MockNonDeferredTool;

    #[async_trait::async_trait]
    impl Tool for MockNonDeferredTool {
        fn name(&self) -> &str {
            "MockNonDeferred"
        }
        fn description(&self) -> &str {
            "A mock non-deferred tool"
        }
        fn input_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": { "cmd": { "type": "string" } },
                "required": ["cmd"]
            })
        }
        fn is_concurrency_safe(&self, _input: &serde_json::Value) -> bool {
            true
        }
        async fn execute(&self, input: serde_json::Value) -> aion_types::tool::ToolResult {
            if input.get("cmd").is_none() {
                return aion_types::tool::ToolResult {
                    content: "Missing cmd".to_string(),
                    is_error: true,
                };
            }
            aion_types::tool::ToolResult {
                content: "ok".to_string(),
                is_error: false,
            }
        }
        fn category(&self) -> aion_protocol::events::ToolCategory {
            aion_protocol::events::ToolCategory::Exec
        }
    }

    fn make_registry_with_deferred() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockDeferredTool {
            schema: json!({
                "type": "object",
                "properties": { "tasks": { "type": "array" } },
                "required": ["tasks"]
            }),
        }));
        registry.register(Box::new(MockNonDeferredTool));
        registry
    }

    #[tokio::test]
    async fn execute_single_deferred_tool_error_missing_required_appends_hint() {
        let registry = make_registry_with_deferred();
        let call = ContentBlock::ToolUse {
            id: "call_1".into(),
            name: "MockDeferred".into(),
            input: json!({}),
            extra: None,
        };
        let (result, _, follow_up_blocks) =
            execute_single(&registry, &call, None, aion_compact::CompactLevel::Off, false).await;
        assert!(follow_up_blocks.is_empty());
        if let ContentBlock::ToolResult { content, is_error, .. } = &result {
            assert!(is_error);
            assert!(content.contains("Missing or invalid 'tasks' array"));
            assert!(content.contains("ToolSearch"));
        } else {
            panic!("expected ToolResult");
        }
    }

    #[tokio::test]
    async fn execute_single_deferred_tool_error_with_required_present_no_hint() {
        let registry = make_registry_with_deferred();
        // tasks is present but wrong type — tool still fails, but required field exists
        let call = ContentBlock::ToolUse {
            id: "call_2".into(),
            name: "MockDeferred".into(),
            input: json!({"tasks": "not_an_array"}),
            extra: None,
        };
        let (result, _, follow_up_blocks) =
            execute_single(&registry, &call, None, aion_compact::CompactLevel::Off, false).await;
        assert!(follow_up_blocks.is_empty());
        if let ContentBlock::ToolResult { content, is_error, .. } = &result {
            // Tool succeeds because input.get("tasks") is Some
            assert!(!is_error);
            assert!(!content.contains("ToolSearch"));
        } else {
            panic!("expected ToolResult");
        }
    }

    #[tokio::test]
    async fn execute_single_deferred_tool_success_no_hint() {
        let registry = make_registry_with_deferred();
        let call = ContentBlock::ToolUse {
            id: "call_3".into(),
            name: "MockDeferred".into(),
            input: json!({"tasks": [{"name": "t1", "prompt": "do x"}]}),
            extra: None,
        };
        let (result, _, follow_up_blocks) =
            execute_single(&registry, &call, None, aion_compact::CompactLevel::Off, false).await;
        assert!(follow_up_blocks.is_empty());
        if let ContentBlock::ToolResult { content, is_error, .. } = &result {
            assert!(!is_error);
            assert_eq!(content, "ok");
        } else {
            panic!("expected ToolResult");
        }
    }

    #[tokio::test]
    async fn execute_single_non_deferred_tool_error_no_hint() {
        let registry = make_registry_with_deferred();
        let call = ContentBlock::ToolUse {
            id: "call_4".into(),
            name: "MockNonDeferred".into(),
            input: json!({}),
            extra: None,
        };
        let (result, _, follow_up_blocks) =
            execute_single(&registry, &call, None, aion_compact::CompactLevel::Off, false).await;
        assert!(follow_up_blocks.is_empty());
        if let ContentBlock::ToolResult { content, is_error, .. } = &result {
            assert!(is_error);
            assert!(content.contains("Missing cmd"));
            assert!(!content.contains("ToolSearch"));
        } else {
            panic!("expected ToolResult");
        }
    }
}

mod sensitive_approval_tests {
    use super::*;
    use aion_tools::Tool;
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct SecretFixture(Arc<AtomicUsize>);
    #[async_trait]
    impl Tool for SecretFixture {
        fn name(&self) -> &str {
            "ellmsha_global_secret_read"
        }
        fn description(&self) -> &str {
            "Synthetic fixture"
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        fn is_concurrency_safe(&self, _: &Value) -> bool {
            false
        }
        fn category(&self) -> ToolCategory {
            ToolCategory::Mcp
        }
        fn sensitive(&self) -> bool {
            true
        }
        fn mcp_approval_identity(&self) -> Option<aion_protocol::McpApprovalIdentity> {
            Some(aion_protocol::McpApprovalIdentity {
                server: "ellmsha-secrets".into(),
                tool: self.name().into(),
            })
        }
        async fn execute(&self, input: Value) -> ToolResult {
            assert_eq!(input["fixture"], "SYNTHETIC_INPUT");
            self.0.fetch_add(1, Ordering::SeqCst);
            ToolResult {
                content: "SYNTHETIC_RAW_RESULT".into(),
                is_error: false,
            }
        }
    }
    struct Emitter {
        manager: Arc<ToolApprovalManager>,
        events: Mutex<Vec<String>>,
        deny: bool,
    }
    impl ProtocolEmitter for Emitter {
        fn emit(&self, event: &ProtocolEvent) -> std::io::Result<()> {
            self.events.lock().unwrap().push(serde_json::to_string(event).unwrap());
            if let ProtocolEvent::ToolRequest { call_id, tool, .. } = event {
                assert_eq!(tool.name, "ellmsha_global_secret_read");
                if self.deny {
                    self.manager.resolve(
                        call_id,
                        ToolApprovalResult::Denied {
                            reason: "Synthetic denial".into(),
                        },
                    );
                } else {
                    self.manager
                        .approve_checked(call_id, aion_protocol::commands::ApprovalScope::Session)
                        .unwrap();
                }
            }
            Ok(())
        }
    }
    #[tokio::test]
    async fn sensitive_approved_raw_result_and_original_identity_are_separate_from_display() {
        for deny in [false, true] {
            let executions = Arc::new(AtomicUsize::new(0));
            let mut registry = ToolRegistry::new();
            registry.register(Box::new(SecretFixture(executions.clone())));
            let manager = Arc::new(ToolApprovalManager::new());
            let emitter = Arc::new(Emitter {
                manager: manager.clone(),
                events: Mutex::new(vec![]),
                deny,
            });
            let writer: Arc<dyn ProtocolEmitter> = emitter.clone();
            let calls = vec![ContentBlock::ToolUse {
                id: "fixture-call".into(),
                name: "ellmsha_global_secret_read".into(),
                input: json!({"fixture":"SYNTHETIC_INPUT"}),
                extra: None,
            }];
            let outcome = execute_tool_calls_with_approval(
                &registry,
                &calls,
                &manager,
                &writer,
                "message",
                None,
                false,
                &[],
                None,
                aion_compact::CompactLevel::Off,
                false,
            )
            .await
            .unwrap();
            let ContentBlock::ToolResult { content, is_error, .. } = &outcome.results[0] else {
                panic!("missing tool result")
            };
            assert_eq!(*is_error, deny);
            assert_eq!(executions.load(Ordering::SeqCst), usize::from(!deny));
            if !deny {
                assert_eq!(content, "SYNTHETIC_RAW_RESULT");
                assert!(manager.is_mcp_approved(&aion_protocol::McpApprovalIdentity {
                    server: "ellmsha-secrets".into(),
                    tool: "ellmsha_global_secret_read".into()
                }));
            }
            let display = emitter.events.lock().unwrap().join("\n");
            assert!(!display.contains("SYNTHETIC_INPUT"));
            assert!(!display.contains("SYNTHETIC_RAW_RESULT"));
            assert!(display.contains("ellmsha_global_secret_read"));
        }
    }
}

mod sensitive_proxy_tests {
    use super::*;
    use aion_mcp::{
        config::{McpServerConfig, TransportType},
        manager::McpManager,
        tool_proxy::register_mcp_tools,
    };
    use aion_protocol::{
        McpApprovalIdentity, McpApprovalPersistence,
        commands::{ApprovalScope, SessionMode},
    };
    use serde_json::{Value, json};
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };
    const TOOL: &str = "ellmsha_global_secret_read";
    #[derive(Default)]
    struct Grants(Mutex<HashSet<McpApprovalIdentity>>);
    impl McpApprovalPersistence for Grants {
        fn supports(&self, id: &McpApprovalIdentity) -> bool {
            id.server == "ellmsha-secrets"
        }
        fn allows(&self, id: &McpApprovalIdentity) -> Result<bool, String> {
            Ok(self.0.lock().unwrap().contains(id))
        }
        fn grant(&self, id: &McpApprovalIdentity) -> Result<(), String> {
            self.0.lock().unwrap().insert(id.clone());
            Ok(())
        }
    }
    struct Emitter {
        manager: Arc<ToolApprovalManager>,
        scope: ApprovalScope,
        events: Mutex<Vec<String>>,
        requests: AtomicUsize,
    }
    impl ProtocolEmitter for Emitter {
        fn emit(&self, event: &ProtocolEvent) -> std::io::Result<()> {
            self.events.lock().unwrap().push(serde_json::to_string(event).unwrap());
            if let ProtocolEvent::ToolRequest { call_id, tool, .. } = event {
                assert!(tool.name.ends_with(TOOL));
                self.requests.fetch_add(1, Ordering::SeqCst);
                self.manager
                    .approve_checked(
                        call_id,
                        match self.scope {
                            ApprovalScope::Once => ApprovalScope::Once,
                            ApprovalScope::Session => ApprovalScope::Session,
                            ApprovalScope::Always => ApprovalScope::Always,
                        },
                    )
                    .unwrap();
            }
            Ok(())
        }
    }
    async fn request_json(socket: &mut tokio::net::TcpStream) -> Option<Value> {
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        let end = loop {
            let n = socket.read(&mut chunk).await.unwrap();
            if n == 0 {
                return None;
            }
            bytes.extend_from_slice(&chunk[..n]);
            if let Some(p) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                break p + 4;
            }
            assert!(bytes.len() < 65536)
        };
        let header = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
        let len = header
            .lines()
            .find_map(|l| l.strip_prefix("content-length:"))
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        while bytes.len() < end + len {
            let n = socket.read(&mut chunk).await.unwrap();
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..n]);
        }
        serde_json::from_slice(&bytes[end..]).ok()
    }
    #[tokio::test]
    async fn production_sensitive_proxy_preserves_raw_arguments_and_aion_approval_modes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let executions = Arc::new(AtomicUsize::new(0));
        let invoked = executions.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = request_json(&mut socket).await;
                let result=request.as_ref().and_then(|r|match r["method"].as_str(){
                Some("initialize")=>Some(json!({"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"synthetic-secrets","version":"1"}})),
                Some("tools/list")=>Some(json!({"tools":[{"name":TOOL,"description":"Synthetic secret read","inputSchema":{"type":"object","properties":{"fixture":{"type":"string"}},"required":["fixture"]}}]})),
                Some("tools/call")=>{assert_eq!(r["params"],json!({"name":TOOL,"arguments":{"fixture":"SYNTHETIC_PROXY_INPUT"}}));invoked.fetch_add(1,Ordering::SeqCst);Some(json!({"content":[{"type":"text","text":"SYNTHETIC_PROXY_RESULT"}],"isError":false}))},_=>None});
                let reply = if let Some(result) = result {
                    let body = json!({"jsonrpc":"2.0","id":request.unwrap()["id"],"result":result}).to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                } else {
                    "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
                };
                socket.write_all(reply.as_bytes()).await.unwrap();
            }
        });
        let configs = HashMap::from([(
            "ellmsha-secrets".into(),
            McpServerConfig {
                transport: TransportType::StreamableHttp,
                command: None,
                args: None,
                env: None,
                url: Some(format!("http://{address}/mcp")),
                headers: None,
                deferred: Some(true),
                startup_timeout_ms: Some(5000),
                sensitive: true,
            },
        )]);
        let mcp = Arc::new(McpManager::connect_all(&configs).await.unwrap());
        let mut registry = ToolRegistry::new();
        // A name collision exercises the production prefixed display name while approvals retain upstream identity.
        register_mcp_tools(&mut registry, &mcp, &[TOOL.into()], &configs);
        let display = format!("mcp__ellmsha-secrets_{TOOL}");
        assert!(registry.get(&display).unwrap().sensitive());
        assert!(registry.get(&display).unwrap().is_deferred());
        let call = ContentBlock::ToolUse {
            id: "synthetic-proxy".into(),
            name: display,
            input: json!({"fixture":"SYNTHETIC_PROXY_INPUT"}),
            extra: None,
        };
        let identity = McpApprovalIdentity {
            server: "ellmsha-secrets".into(),
            tool: TOOL.into(),
        };
        for mode in [SessionMode::Default, SessionMode::AutoEdit, SessionMode::Yolo] {
            for scope in [ApprovalScope::Once, ApprovalScope::Session, ApprovalScope::Always] {
                let grants = Arc::new(Grants::default());
                let manager = Arc::new(ToolApprovalManager::new().with_persistence(grants.clone()));
                manager.set_mode(match mode {
                    SessionMode::Default => SessionMode::Default,
                    SessionMode::AutoEdit => SessionMode::AutoEdit,
                    SessionMode::Yolo => SessionMode::Yolo,
                });
                let emitter = Arc::new(Emitter {
                    manager: manager.clone(),
                    scope: match scope {
                        ApprovalScope::Once => ApprovalScope::Once,
                        ApprovalScope::Session => ApprovalScope::Session,
                        ApprovalScope::Always => ApprovalScope::Always,
                    },
                    events: Mutex::new(vec![]),
                    requests: AtomicUsize::new(0),
                });
                let writer: Arc<dyn ProtocolEmitter> = emitter.clone();
                for _ in 0..2 {
                    let result = execute_tool_calls_with_approval(
                        &registry,
                        std::slice::from_ref(&call),
                        &manager,
                        &writer,
                        "message",
                        None,
                        false,
                        &[],
                        None,
                        aion_compact::CompactLevel::Off,
                        false,
                    )
                    .await
                    .unwrap();
                    let ContentBlock::ToolResult { content, is_error, .. } = &result.results[0] else {
                        panic!("result missing")
                    };
                    assert!(!is_error);
                    assert_eq!(content, "SYNTHETIC_PROXY_RESULT");
                }
                let expected = if matches!(mode, SessionMode::Yolo) {
                    0
                } else if matches!(scope, ApprovalScope::Once) {
                    2
                } else {
                    1
                };
                assert_eq!(emitter.requests.load(Ordering::SeqCst), expected);
                if matches!(scope, ApprovalScope::Always) && !matches!(mode, SessionMode::Yolo) {
                    assert!(manager.is_mcp_approved(&identity));
                    grants.0.lock().unwrap().clear();
                    assert!(!manager.is_mcp_approved(&identity));
                }
                let events = emitter.events.lock().unwrap().join("\n");
                assert!(!events.contains("SYNTHETIC_PROXY_INPUT"));
                assert!(!events.contains("SYNTHETIC_PROXY_RESULT"));
            }
        }
        assert_eq!(executions.load(Ordering::SeqCst), 18);
        mcp.shutdown().await;
        server.abort();
    }
}

#[test]
fn sensitive_summary_keeps_only_structural_target_metadata() {
    let input = serde_json::json!({"id":"019a0000-0000-7000-8000-000000000001","expected_revision":3,"candidate_ids":["private handle", "second"],"title":"SYNTHETIC_RAW","fields":{"password":"SYNTHETIC_RAW"},"query":"SYNTHETIC_RAW"});
    let summary = super::sensitive_summary(&input);
    assert_eq!(summary["record_id"], input["id"]);
    assert_eq!(summary["candidate_count"], 2);
    assert_eq!(summary["expected_revision"], 3);
    assert!(!summary.to_string().contains("SYNTHETIC_RAW"));
    assert!(!summary.to_string().contains("private handle"));
    assert!(
        super::sensitive_summary(&serde_json::json!({"id":"SYNTHETIC_RAW"}))
            .get("record_id")
            .is_none()
    );
}
