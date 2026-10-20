//! A host guidance failure must stop both native child paths before inference.
mod common;

use aion_agent::spawner::{AgentSpawner, ChildPromptPreflight};
use aion_agent::tool_policy::ToolPolicy;
use aion_providers::{LlmProvider, ProviderError};
use aion_types::llm::{LlmEvent, LlmRequest};
use aion_types::spawner::{ForkOverrides, Spawner, SubAgentConfig};
use async_trait::async_trait;
use common::{MockLlmProvider, test_config};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

struct Rules {
    fail: bool,
}
#[async_trait]
impl ChildPromptPreflight for Rules {
    async fn instructions(&self) -> Result<String, String> {
        if self.fail {
            Err("Required project rules unavailable; task not sent".into())
        } else {
            Ok("<ellmsha_required_rules>FULL_PROJECT_RULE</ellmsha_required_rules>".into())
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
async fn both_child_paths_require_rules_before_the_task_and_never_infer_after_failure() {
    for fork in [false, true] {
        for fail in [false, true] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let provider = Arc::new(Capture {
                inner: MockLlmProvider::with_text_response("done"),
                requests: requests.clone(),
            });
            let workspace = tempfile::tempdir().unwrap();
            let spawner = AgentSpawner::new(
                provider,
                test_config(),
                workspace.path().to_owned(),
                ToolPolicy::allow_only(["Read"]),
            )
            .with_child_preflight(Arc::new(Rules { fail }));
            let task = SubAgentConfig {
                name: "qa".into(),
                prompt: "ACTUAL_CHILD_TASK".into(),
                system_prompt: Some("CUSTOM_CHILD_SYSTEM".into()),
                max_turns: 1,
                max_tokens: 128,
            };
            let result = if fork {
                spawner.spawn_fork(task, ForkOverrides::default()).await
            } else {
                spawner.spawn_one(task).await
            };
            assert_eq!(result.is_error, fail, "fork={fork}, fail={fail}: {}", result.text);
            let captured = requests.lock().unwrap();
            if fail {
                assert!(captured.is_empty());
                assert_eq!(result.turns, 0);
            } else {
                assert_eq!(captured.len(), 1);
                assert!(
                    captured[0].system.find("FULL_PROJECT_RULE").unwrap()
                        < captured[0].system.find("CUSTOM_CHILD_SYSTEM").unwrap()
                );
                let payload = serde_json::to_string(&captured[0].messages).unwrap();
                assert!(payload.find("FULL_PROJECT_RULE").unwrap() < payload.find("ACTUAL_CHILD_TASK").unwrap());
            }
        }
    }
}
