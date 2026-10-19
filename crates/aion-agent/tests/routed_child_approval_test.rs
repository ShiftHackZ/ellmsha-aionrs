mod common;
use aion_agent::{
    spawner::{AgentSpawner, ChildProvider, ChildRouter, SubAgentConfig},
    tool_policy::ToolPolicy,
};
use aion_protocol::commands::SessionMode;
use aion_protocol::{ToolApprovalManager, ToolApprovalResult, events::ProtocolEvent, writer::ProtocolEmitter};
use common::{MockLlmProvider, test_config};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Router {
    calls: Arc<AtomicUsize>,
}
#[async_trait::async_trait]
impl ChildRouter for Router {
    async fn resolve(&self, prompt: &str) -> Result<ChildProvider, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ChildProvider {
            provider: Arc::new(MockLlmProvider::with_turns(vec![
                vec![
                    aion_types::llm::LlmEvent::ToolUse {
                        id: "same-call".into(),
                        name: "Write".into(),
                        input: serde_json::json!({"file_path":prompt,"content":"approved child"}),
                        extra: None,
                    },
                    aion_types::llm::LlmEvent::Done {
                        stop_reason: aion_types::message::StopReason::ToolUse,
                        usage: Default::default(),
                    },
                ],
                vec![
                    aion_types::llm::LlmEvent::TextDelta("Child completed its permitted work".into()),
                    aion_types::llm::LlmEvent::Done {
                        stop_reason: aion_types::message::StopReason::EndTurn,
                        usage: Default::default(),
                    },
                ],
            ])),
            config: test_config(),
            decision: "fixture-child".into(),
        })
    }
}
struct Review {
    manager: Arc<ToolApprovalManager>,
    ids: Mutex<Vec<String>>,
}
impl ProtocolEmitter for Review {
    fn emit(&self, event: &ProtocolEvent) -> std::io::Result<()> {
        if let ProtocolEvent::ToolRequest { call_id, tool, .. } = event {
            self.ids.lock().unwrap().push(call_id.clone());
            let allowed = serde_json::to_string(tool).unwrap().contains("allow.txt");
            self.manager.resolve(
                call_id,
                if allowed {
                    ToolApprovalResult::Approved
                } else {
                    ToolApprovalResult::Denied {
                        reason: "parent rejected".into(),
                    }
                },
            );
        }
        Ok(())
    }
}
fn task(path: &std::path::Path) -> SubAgentConfig {
    SubAgentConfig {
        name: "child".into(),
        prompt: path.to_string_lossy().into_owned(),
        max_turns: 3,
        max_tokens: 256,
        system_prompt: None,
    }
}
fn spawner(path: &std::path::Path, calls: Arc<AtomicUsize>) -> AgentSpawner {
    let mut config = test_config();
    config.tools.auto_approve = true; // The stale initial yolo flag must never bypass the live manager.
    AgentSpawner::new(
        Arc::new(MockLlmProvider::with_text_response("unused")),
        config,
        path.into(),
        ToolPolicy::Unrestricted,
    )
    .with_child_router(Arc::new(Router { calls }))
}
#[tokio::test]
async fn routed_children_share_live_parent_approval_and_namespace_colliding_calls() {
    let dir = tempfile::tempdir().unwrap();
    let manager = Arc::new(ToolApprovalManager::new());
    manager.set_mode(SessionMode::Yolo);
    let review = Arc::new(Review {
        manager: manager.clone(),
        ids: Mutex::new(Vec::new()),
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let spawner = spawner(dir.path(), calls.clone()).with_host_approval(manager.clone(), review.clone());
    manager.set_mode(SessionMode::Default);
    let allow = dir.path().join("allow.txt");
    let deny = dir.path().join("deny.txt");
    let results = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        spawner.spawn_parallel(vec![task(&allow), task(&deny)]),
    )
    .await
    .unwrap();
    assert!(results.iter().all(|r| !r.is_error), "{results:?}");
    assert_eq!(std::fs::read_to_string(allow).unwrap(), "approved child");
    assert!(!deny.exists());
    let ids = review.ids.lock().unwrap();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    assert!(ids.iter().all(|id| id.ends_with(":same-call")));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn routed_child_plan_mode_denies_mutation_even_with_yolo_approval() {
    let dir = tempfile::tempdir().unwrap();
    let manager = Arc::new(ToolApprovalManager::new());
    manager.set_mode(SessionMode::Yolo);
    let review = Arc::new(Review {
        manager: manager.clone(),
        ids: Mutex::new(Vec::new()),
    });
    let spawner = spawner(dir.path(), Arc::new(AtomicUsize::new(0)))
        .with_host_approval(manager, review.clone())
        .with_parent_plan(Arc::new(AtomicBool::new(true)));
    let path = dir.path().join("allow.txt");
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), spawner.spawn_one(task(&path)))
        .await
        .unwrap();
    assert!(!result.is_error, "{result:?}");
    assert!(!path.exists());
    assert!(review.ids.lock().unwrap().is_empty());
}
#[tokio::test]
async fn routed_child_without_parent_approval_fails_before_model_resolution() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let result = spawner(dir.path(), calls.clone())
        .spawn_one(task(&dir.path().join("allow.txt")))
        .await;
    assert!(result.is_error);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn ordinary_child_keeps_existing_auto_approval_without_inheriting_router_plan_policy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ordinary.txt");
    let provider = Router {
        calls: Arc::new(AtomicUsize::new(0)),
    }
    .resolve(path.to_str().unwrap())
    .await
    .unwrap()
    .provider;
    let spawner = AgentSpawner::new(provider, test_config(), dir.path().into(), ToolPolicy::Unrestricted)
        .with_parent_plan(Arc::new(AtomicBool::new(true)));
    let result = spawner.spawn_one(task(&path)).await;
    assert!(!result.is_error, "{result:?}");
    assert_eq!(std::fs::read_to_string(path).unwrap(), "approved child");
    assert!(!result.text.contains("Auto Router decision"));
}

struct PanickingProvider;
#[async_trait::async_trait]
impl aion_providers::LlmProvider for PanickingProvider {
    async fn stream(
        &self,
        _request: &aion_types::llm::LlmRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<aion_types::llm::LlmEvent>, aion_providers::ProviderError> {
        panic!("ordinary child panic fixture");
    }
}

#[tokio::test]
async fn ordinary_parallel_children_preserve_panic_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let spawner = AgentSpawner::new(
        Arc::new(PanickingProvider),
        test_config(),
        dir.path().into(),
        ToolPolicy::Unrestricted,
    );
    let results = spawner.spawn_parallel(vec![task(dir.path()), task(dir.path())]).await;
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.is_error && r.text.contains("Task join error")));
}

struct MixedRouter;
#[async_trait::async_trait]
impl ChildRouter for MixedRouter {
    async fn resolve(&self, prompt: &str) -> Result<ChildProvider, String> {
        Ok(ChildProvider {
            provider: if prompt.ends_with("panic") {
                Arc::new(PanickingProvider)
            } else {
                Arc::new(MockLlmProvider::with_text_response("Other child completed"))
            },
            config: test_config(),
            decision: "mixed-route".into(),
        })
    }
}

#[tokio::test]
async fn routed_parallel_child_panic_does_not_interrupt_parent_or_sibling() {
    let dir = tempfile::tempdir().unwrap();
    let manager = Arc::new(ToolApprovalManager::new());
    let review = Arc::new(Review {
        manager: manager.clone(),
        ids: Mutex::new(Vec::new()),
    });
    let spawner = AgentSpawner::new(
        Arc::new(PanickingProvider),
        test_config(),
        dir.path().into(),
        ToolPolicy::Unrestricted,
    )
    .with_child_router(Arc::new(MixedRouter))
    .with_host_approval(manager, review);
    let results = spawner
        .spawn_parallel(vec![
            task(&dir.path().join("panic")),
            task(&dir.path().join("complete")),
        ])
        .await;
    assert_eq!(results.len(), 2);
    assert!(results[0].is_error && results[0].text.contains("panicked"));
    assert!(!results[1].is_error && results[1].text.contains("Other child completed"));
}

struct PausedProvider {
    inner: MockLlmProvider,
    started: tokio::sync::mpsc::UnboundedSender<()>,
    gate: Arc<tokio::sync::Notify>,
}
#[async_trait::async_trait]
impl aion_providers::LlmProvider for PausedProvider {
    async fn stream(
        &self,
        request: &aion_types::llm::LlmRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<aion_types::llm::LlmEvent>, aion_providers::ProviderError> {
        self.started.send(()).unwrap();
        self.gate.notified().await;
        self.inner.stream(request).await
    }
}
struct PausedRouter {
    started: tokio::sync::mpsc::UnboundedSender<()>,
    gate: Arc<tokio::sync::Notify>,
}
#[async_trait::async_trait]
impl ChildRouter for PausedRouter {
    async fn resolve(&self, prompt: &str) -> Result<ChildProvider, String> {
        Ok(ChildProvider {
            provider: Arc::new(PausedProvider {
                inner: MockLlmProvider::with_tool_use(
                    "after-stop",
                    "Write",
                    serde_json::json!({"file_path":prompt,"content":"must not be written"}),
                ),
                started: self.started.clone(),
                gate: self.gate.clone(),
            }),
            config: test_config(),
            decision: "paused-route".into(),
        })
    }
}
#[tokio::test]
async fn stopping_parent_cancels_parallel_children_before_any_later_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let manager = Arc::new(ToolApprovalManager::new());
    manager.set_mode(SessionMode::Yolo);
    let writer = Arc::new(Review {
        manager: manager.clone(),
        ids: Mutex::new(Vec::new()),
    });
    let (started, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let gate = Arc::new(tokio::sync::Notify::new());
    let spawner = AgentSpawner::new(
        Arc::new(MockLlmProvider::with_text_response("unused")),
        test_config(),
        dir.path().into(),
        ToolPolicy::Unrestricted,
    )
    .with_child_router(Arc::new(PausedRouter {
        started,
        gate: gate.clone(),
    }))
    .with_host_approval(manager, writer);
    let first = dir.path().join("first.txt");
    let second = dir.path().join("second.txt");
    let tasks = vec![task(&first), task(&second)];
    let parent = tokio::spawn(async move { spawner.spawn_parallel(tasks).await });
    for _ in 0..2 {
        tokio::time::timeout(std::time::Duration::from_secs(5), receiver.recv())
            .await
            .unwrap()
            .unwrap();
    }
    parent.abort();
    assert!(parent.await.unwrap_err().is_cancelled());
    gate.notify_waiters();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!first.exists());
    assert!(!second.exists());
}
