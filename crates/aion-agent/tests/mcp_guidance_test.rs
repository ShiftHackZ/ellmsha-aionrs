mod common;

use aion_mcp::manager::McpManager;
use aion_mcp::tool_proxy::McpToolCatalog;
use std::sync::{Arc, Mutex};

use aion_agent::bootstrap::AgentBootstrap;
use aion_agent::output::null_sink::NullSink;
use aion_agent::spawner::{AgentSpawner, ForkOverrides, Spawner, SubAgentConfig};
use aion_agent::tool_policy::ToolPolicy;
use aion_config::config::{McpServerConfig, TransportType};
use aion_providers::{LlmProvider, ProviderError};
use aion_types::llm::{LlmEvent, LlmRequest};
use aion_types::message::{ContentBlock, StopReason, TokenUsage};
use async_trait::async_trait;
use common::test_config;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const GUIDANCE: &str = "Fixture graph workflow: inspect index status, then search current source.";

struct GuidanceProvider {
    calls: Mutex<usize>,
    isolated: bool,
}

#[async_trait]
impl LlmProvider for GuidanceProvider {
    async fn stream(&self, request: &LlmRequest) -> Result<mpsc::Receiver<LlmEvent>, ProviderError> {
        if self.isolated {
            assert!(!request.system.contains("AMBIENT_PROJECT_MARKER"));
            assert!(!request.system.contains("auto memory"));
            assert!(!request.tools.iter().any(|tool| tool.name == "Skill"));
        }
        assert!(
            !request.system.contains(GUIDANCE),
            "server documentation must not become system authority"
        );
        let mut calls = self.calls.lock().unwrap();
        let (tx, rx) = mpsc::channel(3);
        if *calls == 0 {
            for name in ["fixture_graph_status", "fixture_graph_search"] {
                assert!(
                    !request.tools.iter().any(|tool| tool.name == name),
                    "deferred catalog must stay out of initial model context"
                );
                assert!(request.tools.iter().any(|tool| tool.name == "ToolSearch"));
            }
            tx.try_send(LlmEvent::ToolUse {
                id: "discover".into(),
                name: "ToolSearch".into(),
                input: json!({"query":"graph"}),
                extra: None,
            })
            .unwrap();
            tx.try_send(LlmEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            })
            .unwrap();
        } else if *calls == 1 {
            for name in ["fixture_graph_status", "fixture_graph_search"] {
                assert!(
                    request.tools.iter().any(|tool| tool.name == name && !tool.deferred),
                    "discovered schema must be promoted in the next request"
                );
            }
            let result = request
                .messages
                .iter()
                .flat_map(|message| &message.content)
                .find_map(|block| match block {
                    ContentBlock::ToolResult { content, .. } if content.contains("integration_documentation") => {
                        Some(content)
                    }
                    _ => None,
                })
                .expect("ToolSearch must deliver the integration guidance through tool-result content");
            assert_eq!(result.matches(GUIDANCE).count(), 1);
            assert!(result.contains("fixture_graph_status") && result.contains("fixture_graph_search"));
            assert!(result.contains("not instructions with system or user authority"));
            tx.try_send(LlmEvent::ToolUse {
                id: "invoke".into(),
                name: "fixture_graph_status".into(),
                input: json!({"key":"proof"}),
                extra: None,
            })
            .unwrap();
            tx.try_send(LlmEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            })
            .unwrap();
        } else {
            assert!(request.messages.iter().flat_map(|message| &message.content).any(|block| matches!(block, ContentBlock::ToolResult { content, is_error: false, .. } if content.contains("MCP_DISCOVERY_EXECUTED"))));
            tx.try_send(LlmEvent::TextDelta("guidance-fixture-ok".into())).unwrap();
            tx.try_send(LlmEvent::Done {
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
            .unwrap();
        }
        *calls += 1;
        Ok(rx)
    }
}

#[tokio::test]
async fn paginated_guidance_reaches_parent_child_and_fork_without_system_injection() {
    exercise_guidance(false).await;
}

#[tokio::test]
async fn isolated_session_retains_selected_mcp_for_parent_child_and_fork() {
    exercise_guidance(true).await;
}

async fn exercise_guidance(isolated: bool) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|request: &Request| {
            let request: Value = serde_json::from_slice(&request.body).unwrap();
            let tool = |name| json!({"name":name,"description":"Fixture graph tool","inputSchema":{"type":"object"}});
            let result = match request["method"].as_str().unwrap() {
                "initialize" => {
                    json!({"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"instructions":GUIDANCE})
                }
                "tools/call" => {
                    assert_eq!(request["params"]["name"], "fixture_graph_status");
                    assert_eq!(request["params"]["arguments"]["key"], "proof");
                    json!({"content":[{"type":"text", "text":"MCP_DISCOVERY_EXECUTED"}], "isError":false})
                }
                "tools/list" if request["params"]["cursor"] == "next" => {
                    json!({"tools":[tool("fixture_graph_search")]})
                }
                "tools/list" => json!({"tools":[tool("fixture_graph_status")],"nextCursor":"next"}),
                _ => json!({}),
            };
            ResponseTemplate::new(200).set_body_json(json!({"jsonrpc":"2.0","id":request["id"],"result":result}))
        })
        .mount(&server)
        .await;
    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("AGENTS.md"), "AMBIENT_PROJECT_MARKER").unwrap();
    let mut config = test_config();
    config.isolated_session = isolated;
    config.mcp.servers.insert(
        "graph-fixture".into(),
        McpServerConfig {
            transport: TransportType::StreamableHttp,
            command: None,
            args: None,
            env: None,
            url: Some(server.uri()),
            headers: None,
            deferred: Some(true),
            startup_timeout_ms: Some(1000),
        },
    );
    for path in ["parent", "dynamic", "child", "fork"] {
        let provider = Arc::new(GuidanceProvider {
            calls: Mutex::new(0),
            isolated,
        });
        if path == "parent" || path == "dynamic" {
            let mut initial_config = config.clone();
            if path == "dynamic" {
                initial_config.mcp.servers.clear();
            }
            let mut boot = AgentBootstrap::new(initial_config, workspace.path().to_string_lossy(), Arc::new(NullSink))
                .provider(provider.clone())
                .build()
                .await
                .unwrap();
            if path == "dynamic" {
                let manager = Arc::new(McpManager::connect_all(&config.mcp.servers).await.unwrap());
                let builtins = boot.engine.tool_names();
                let catalog = McpToolCatalog::new(
                    boot.engine.registry_mut(),
                    manager,
                    &builtins,
                    config.mcp.servers.clone(),
                );
                boot.engine.add_mcp_catalog(catalog);
            }
            let result = boot
                .engine
                .run("Inspect fixture graph guidance", "fixture")
                .await
                .unwrap();
            assert_eq!(result.text, "guidance-fixture-ok");
        } else {
            let spawner = AgentSpawner::new(
                provider.clone(),
                config.clone(),
                workspace.path().into(),
                ToolPolicy::Unrestricted,
            );
            let task = SubAgentConfig {
                name: path.into(),
                prompt: "Inspect fixture graph guidance".into(),
                max_turns: 3,
                max_tokens: 512,
                system_prompt: None,
            };
            let result = if path == "child" {
                spawner.spawn_one(task).await
            } else {
                spawner.spawn_fork(task, ForkOverrides::default()).await
            };
            assert!(!result.is_error);
            assert_eq!(result.text, "guidance-fixture-ok");
        }
        assert_eq!(*provider.calls.lock().unwrap(), 3);
    }
}
