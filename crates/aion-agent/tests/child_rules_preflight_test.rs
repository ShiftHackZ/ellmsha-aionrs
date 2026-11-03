//! Native children receive complete fresh guidance exactly once. A host
//! guidance failure must stop both paths before routing or inference.
mod common;

use aion_agent::spawner::{AgentSpawner, ChildPromptPreflight, ChildProvider, ChildRouter};
use aion_agent::tool_policy::ToolPolicy;
use aion_providers::{LlmProvider, ProviderError};
use aion_types::llm::{LlmEvent, LlmRequest};
use aion_types::message::{StopReason, TokenUsage};
use aion_types::spawner::{ForkOverrides, Spawner, SubAgentConfig};
use async_trait::async_trait;
use common::{MockLlmProvider, test_config};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

struct Rules {
    fail: bool,
    calls: AtomicUsize,
}
#[async_trait]
impl ChildPromptPreflight for Rules {
    async fn instructions(&self) -> Result<String, String> {
        let revision = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail {
            Err("Required project rules unavailable; task not sent".into())
        } else {
            Ok(format!(
                "<ellmsha_required_rules>FULL_GLOBAL_RULE\nFULL_PROJECT_RULE_REVISION_{revision}</ellmsha_required_rules>"
            ))
        }
    }
}
struct Capture {
    inner: MockLlmProvider,
    requests: Arc<Mutex<Vec<LlmRequest>>>,
}
#[async_trait]
impl LlmProvider for Capture {
    async fn stream(&self, request: &LlmRequest) -> Result<mpsc::Receiver<LlmEvent>, ProviderError> {
        self.requests.lock().unwrap().push(request.clone());
        self.inner.stream(request).await
    }
}
#[tokio::test]
async fn both_child_paths_include_fresh_complete_rules_once_and_preserve_the_system_prompt() {
    for fork in [false, true] {
        for custom_system in [false, true] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let workspace = tempfile::tempdir().unwrap();
            let fixture = workspace.path().join("fixture.txt");
            std::fs::write(&fixture, "ISOLATED_TOOL_RESULT").unwrap();
            let tool_round = vec![
                LlmEvent::ToolUse {
                    id: "read-fixture".into(),
                    name: "Read".into(),
                    input: json!({"file_path": fixture}),
                    extra: None,
                },
                LlmEvent::Done {
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                },
            ];
            let final_round = vec![
                LlmEvent::TextDelta("done".into()),
                LlmEvent::Done {
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
            ];
            let provider = Arc::new(Capture {
                inner: MockLlmProvider::with_turns(vec![
                    tool_round.clone(),
                    final_round.clone(),
                    tool_round,
                    final_round,
                ]),
                requests: requests.clone(),
            });
            let mut config = test_config();
            config.system_prompt = Some("INHERITED_PARENT_SYSTEM".into());
            let rules = Arc::new(Rules {
                fail: false,
                calls: AtomicUsize::new(0),
            });
            let spawner = AgentSpawner::new(
                provider,
                config,
                workspace.path().to_owned(),
                ToolPolicy::allow_only(["Read"]),
            )
            .with_child_preflight(rules.clone());
            for _ in 0..2 {
                let mut task = child_task();
                task.system_prompt = custom_system.then(|| "CUSTOM_CHILD_SYSTEM".into());
                let result = if fork {
                    spawner.spawn_fork(task, ForkOverrides::default()).await
                } else {
                    spawner.spawn_one(task).await
                };
                assert!(!result.is_error, "fork={fork}, custom={custom_system}: {}", result.text);
                assert_eq!(result.turns, 2);
            }
            assert_eq!(rules.calls.load(Ordering::SeqCst), 2);
            let captured = requests.lock().unwrap();
            assert_eq!(captured.len(), 4);
            let expected_system = if custom_system {
                "CUSTOM_CHILD_SYSTEM"
            } else {
                "INHERITED_PARENT_SYSTEM"
            };
            for (index, request) in captured.iter().enumerate() {
                assert_eq!(request.system, expected_system);
                let payload = serde_json::to_string(&request.messages).unwrap();
                assert_eq!(payload.matches("FULL_GLOBAL_RULE").count(), 1);
                assert_eq!(payload.matches("FULL_PROJECT_RULE_REVISION_").count(), 1);
                let revision = index / 2 + 1;
                assert!(payload.contains(&format!("FULL_PROJECT_RULE_REVISION_{revision}")));
                assert!(payload.find("FULL_PROJECT_RULE").unwrap() < payload.find("ACTUAL_CHILD_TASK").unwrap());
                if index % 2 == 1 {
                    assert!(payload.contains("ISOLATED_TOOL_RESULT"));
                }
            }
        }
    }
}

struct RoutingMustNotStart;
#[async_trait]
impl ChildRouter for RoutingMustNotStart {
    async fn resolve(&self, _: &str) -> Result<ChildProvider, String> {
        panic!("routing must not run when mandatory guidance fails");
    }
}

#[tokio::test]
async fn failed_rules_stop_both_child_paths_before_routing_or_inference() {
    for fork in [false, true] {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(Capture {
            inner: MockLlmProvider::with_text_response("must not run"),
            requests: requests.clone(),
        });
        let workspace = tempfile::tempdir().unwrap();
        let rules = Arc::new(Rules {
            fail: true,
            calls: AtomicUsize::new(0),
        });
        let spawner = AgentSpawner::new(
            provider,
            test_config(),
            workspace.path().to_owned(),
            ToolPolicy::allow_only(["Read"]),
        )
        .with_child_preflight(rules.clone())
        .with_child_router(Arc::new(RoutingMustNotStart));
        let result = if fork {
            spawner.spawn_fork(child_task(), ForkOverrides::default()).await
        } else {
            spawner.spawn_one(child_task()).await
        };
        assert!(result.is_error);
        assert_eq!(result.text, "Required project rules unavailable; task not sent");
        assert_eq!(result.turns, 0);
        assert_eq!(rules.calls.load(Ordering::SeqCst), 1);
        assert!(requests.lock().unwrap().is_empty());
    }
}

fn child_task() -> SubAgentConfig {
    SubAgentConfig {
        name: "qa".into(),
        prompt: "ACTUAL_CHILD_TASK".into(),
        system_prompt: None,
        max_turns: 2,
        max_tokens: 128,
    }
}
