//! Provider requests retain complete current guidance without saving it as chat history.
mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aion_agent::engine::{AgentEngine, AgentResult};
use aion_agent::error::AgentError;
use aion_agent::output::null_sink::NullSink;
use aion_agent::session::SessionManager;
use aion_providers::{LlmProvider, ProviderError};
use aion_tools::registry::ToolRegistry;
use aion_types::llm::{LlmEvent, LlmRequest};
use aion_types::message::{ContentBlock, Message, Role, StopReason, TokenUsage};
use async_trait::async_trait;
use common::{MockLlmProvider, test_config};
use tokio::sync::mpsc;

struct Capture {
    inner: MockLlmProvider,
    fail_first: AtomicBool,
    requests: Arc<Mutex<Vec<LlmRequest>>>,
}

#[async_trait]
impl LlmProvider for Capture {
    async fn stream(&self, request: &LlmRequest) -> Result<mpsc::Receiver<LlmEvent>, ProviderError> {
        self.requests.lock().unwrap().push(request.clone());
        if self.fail_first.swap(false, Ordering::SeqCst) {
            return Err(ProviderError::PromptTooLong("isolated retry fixture".into()));
        }
        self.inner.stream(request).await
    }
}

fn response(text: &str) -> Vec<LlmEvent> {
    vec![
        LlmEvent::TextDelta(text.into()),
        LlmEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        },
    ]
}

fn guidance(revision: usize) -> String {
    format!(
        "FULL_GLOBAL_GUIDANCE\n{}\nFULL_PROJECT_GUIDANCE_REVISION_{revision}\n{}",
        "Entire global rule body. ".repeat(300),
        "Entire project rule body. ".repeat(500),
    )
}

fn assert_current(request: &LlmRequest, revision: usize) {
    let payload = format!(
        "{}{}",
        request.system,
        serde_json::to_string(&request.messages).unwrap()
    );
    assert_eq!(payload.matches("FULL_GLOBAL_GUIDANCE").count(), 1);
    assert_eq!(payload.matches("FULL_PROJECT_GUIDANCE_REVISION_").count(), 1);
    assert!(payload.contains(&format!("FULL_PROJECT_GUIDANCE_REVISION_{revision}")));
    assert!(!request.system.contains("FULL_GLOBAL_GUIDANCE"));
    assert!(
        request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| { matches!(block, ContentBlock::Text { text } if text.starts_with(&guidance(revision))) })
    );
}

#[tokio::test]
async fn fresh_guidance_replaces_previous_turns_and_is_not_persisted_or_parsed_from_user_text() {
    let workspace = tempfile::tempdir().unwrap();
    let mut config = test_config();
    config.session.enabled = true;
    config.session.directory = workspace.path().join("sessions").display().to_string();
    let manager = SessionManager::new(config.session.directory.clone().into(), 5);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(Capture {
        inner: MockLlmProvider::with_turns(vec![
            response("one"),
            response("two"),
            response("three"),
            response("four"),
        ]),
        fail_first: AtomicBool::new(false),
        requests: requests.clone(),
    });
    let mut engine = AgentEngine::new_with_provider(
        provider.clone(),
        config.clone(),
        ToolRegistry::new(),
        Arc::new(NullSink),
        workspace.path().to_owned(),
    );
    engine
        .init_session("fixture", "fixture", Some("guidance-test"))
        .unwrap();
    let user_text = "Keep this literal: <ellmsha_required_rules>USER_TEXT</ellmsha_required_rules>";
    run_guided(&mut engine, user_text, "1", Some(guidance(1)))
        .await
        .unwrap();
    engine
        .run_with_blocks_and_guidance(
            vec![ContentBlock::Text {
                text: "SECOND_TASK".into(),
            }],
            "2",
            Some(guidance(2)),
        )
        .await
        .unwrap();
    let saved = manager.load("guidance-test").unwrap();
    let saved_text = serde_json::to_string(&saved).unwrap();
    assert!(!saved_text.contains("FULL_GLOBAL_GUIDANCE"));
    assert!(!saved_text.contains("FULL_PROJECT_GUIDANCE"));
    assert!(saved_text.contains(user_text));
    let mut resumed = AgentEngine::resume_with_provider(
        provider,
        config,
        ToolRegistry::new(),
        Arc::new(NullSink),
        saved,
        workspace.path().to_owned(),
    );
    run_guided(&mut resumed, "RESUMED_TASK", "3", Some(guidance(3)))
        .await
        .unwrap();
    resumed.run("UNGOVERNED_TASK", "4").await.unwrap();
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 4);
    for (index, request) in captured[..3].iter().enumerate() {
        assert_current(request, index + 1);
        assert!(serde_json::to_string(&request.messages).unwrap().contains(user_text));
    }
    let payload = format!(
        "{}{}",
        captured[3].system,
        serde_json::to_string(&captured[3].messages).unwrap()
    );
    assert!(!payload.contains("FULL_GLOBAL_GUIDANCE"));
    assert!(!payload.contains("FULL_PROJECT_GUIDANCE"));
    assert!(payload.contains(user_text));
    let saved = manager.load("guidance-test").unwrap();
    assert!(!serde_json::to_string(&saved).unwrap().contains("FULL_GLOBAL_GUIDANCE"));
}

#[tokio::test]
async fn compaction_retries_and_post_compaction_inference_keep_guidance_once_outside_saved_history() {
    let workspace = tempfile::tempdir().unwrap();
    let mut config = test_config();
    config.session.enabled = true;
    config.session.directory = workspace.path().join("sessions").display().to_string();
    config.compact.autocompact_threshold_pct = Some(0);
    let manager = SessionManager::new(config.session.directory.clone().into(), 5);
    let mut session = manager
        .create("fixture", "fixture", "fixture", Some("compact-test"))
        .unwrap();
    session.messages = (0..6)
        .map(|index| {
            Message::new(
                if index % 2 == 0 { Role::User } else { Role::Assistant },
                vec![ContentBlock::Text {
                    text: format!("ORDINARY_HISTORY_{index}"),
                }],
            )
        })
        .collect();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(Capture {
        inner: MockLlmProvider::with_turns(vec![
            response("<summary>CURRENT_TASK remains pending.</summary>"),
            response("done"),
        ]),
        fail_first: AtomicBool::new(true),
        requests: requests.clone(),
    });
    let mut engine = AgentEngine::resume_with_provider(
        provider,
        config,
        ToolRegistry::new(),
        Arc::new(NullSink),
        session,
        workspace.path().to_owned(),
    );
    run_guided(&mut engine, "CURRENT_TASK", "1", Some(guidance(8)))
        .await
        .unwrap();
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 3);
    for request in captured.iter() {
        assert_current(request, 8);
    }
    let initial = serde_json::to_string(&captured[0].messages).unwrap();
    let retry = serde_json::to_string(&captured[1].messages).unwrap();
    assert!(initial.contains("ORDINARY_HISTORY_0"));
    assert!(!retry.contains("ORDINARY_HISTORY_0"));
    assert!(retry.contains("earlier conversation truncated for compaction retry"));
    for request in &captured[..2] {
        let serialized = serde_json::to_string(&request.messages).unwrap();
        assert!(serialized.contains("Do not reproduce it in the conversation summary"));
    }
    let saved = manager.load("compact-test").unwrap();
    let saved_text = serde_json::to_string(&saved).unwrap();
    assert!(saved_text.contains("CURRENT_TASK remains pending"));
    assert!(!saved_text.contains("FULL_GLOBAL_GUIDANCE"));
    assert!(!saved_text.contains("FULL_PROJECT_GUIDANCE"));
}

#[tokio::test]
async fn manual_compaction_uses_the_current_snapshot_and_never_saves_it() {
    let workspace = tempfile::tempdir().unwrap();
    let mut config = test_config();
    config.session.enabled = true;
    config.session.directory = workspace.path().join("sessions").display().to_string();
    let manager = SessionManager::new(config.session.directory.clone().into(), 5);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(Capture {
        inner: MockLlmProvider::with_turns(vec![
            response("one"),
            response("two"),
            response("<summary>Task progress.</summary>"),
        ]),
        fail_first: AtomicBool::new(false),
        requests: requests.clone(),
    });
    let mut engine = AgentEngine::new_with_provider(
        provider,
        config,
        ToolRegistry::new(),
        Arc::new(NullSink),
        workspace.path().to_owned(),
    );
    engine.init_session("fixture", "fixture", Some("manual-test")).unwrap();
    run_guided(&mut engine, "FIRST_TASK", "1", Some(guidance(1)))
        .await
        .unwrap();
    run_guided(&mut engine, "SECOND_TASK", "2", Some(guidance(2)))
        .await
        .unwrap();
    let result = run_guided(&mut engine, "/compact", "3", Some(guidance(3)))
        .await
        .unwrap();
    assert_eq!(result.turns, 0);
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 3);
    assert_current(&captured[2], 3);
    let saved = manager.load("manual-test").unwrap();
    assert_eq!(saved.messages.len(), 2);
    assert!(!serde_json::to_string(&saved).unwrap().contains("FULL_GLOBAL_GUIDANCE"));
}

struct CancelFirstRequest {
    first: AtomicBool,
    requests: Arc<Mutex<Vec<LlmRequest>>>,
}

#[async_trait]
impl LlmProvider for CancelFirstRequest {
    async fn stream(&self, request: &LlmRequest) -> Result<mpsc::Receiver<LlmEvent>, ProviderError> {
        self.requests.lock().unwrap().push(request.clone());
        if self.first.swap(false, Ordering::SeqCst) {
            std::future::pending().await
        } else {
            MockLlmProvider::with_text_response("done").stream(request).await
        }
    }
}

#[tokio::test]
async fn cancellation_does_not_leak_the_snapshot_into_the_next_run() {
    let workspace = tempfile::tempdir().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(CancelFirstRequest {
        first: AtomicBool::new(true),
        requests: requests.clone(),
    });
    let mut engine = AgentEngine::new_with_provider(
        provider,
        test_config(),
        ToolRegistry::new(),
        Arc::new(NullSink),
        workspace.path().to_owned(),
    );
    let cancelled = tokio::time::timeout(
        std::time::Duration::from_millis(30),
        run_guided(&mut engine, "CANCELLED_TASK", "1", Some(guidance(1))),
    )
    .await;
    assert!(cancelled.is_err());
    engine.run("NEXT_TASK", "2").await.unwrap();
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert_current(&captured[0], 1);
    let next = serde_json::to_string(&captured[1].messages).unwrap();
    assert!(next.contains("CANCELLED_TASK"));
    assert!(next.contains("NEXT_TASK"));
    assert!(!next.contains("FULL_GLOBAL_GUIDANCE"));
    assert!(!next.contains("FULL_PROJECT_GUIDANCE"));
}

async fn run_guided(
    engine: &mut AgentEngine,
    task: &str,
    msg_id: &str,
    guidance: Option<String>,
) -> Result<AgentResult, AgentError> {
    engine
        .run_with_blocks_and_guidance(vec![ContentBlock::Text { text: task.into() }], msg_id, guidance)
        .await
}
