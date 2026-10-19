use aion_protocol::{ToolApprovalManager, writer::ProtocolEmitter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use async_trait::async_trait;
use futures::FutureExt;

use aion_config::config::Config;
use aion_providers::LlmProvider;
use aion_tools::edit::EditTool;
use aion_tools::exec_command::ExecCommandTool;
use aion_tools::glob::GlobTool;
use aion_tools::grep::GrepTool;
use aion_tools::read::ReadTool;
use aion_tools::registry::ToolRegistry;
use aion_tools::tool_search::ToolSearchTool;
use aion_tools::write::WriteTool;
use aion_types::message::TokenUsage;

use crate::bootstrap::{AgentBootstrap, McpBootstrap};
use crate::engine::AgentEngine;
use crate::output::OutputSink;
use crate::output::null_sink::NullSink;
use crate::tool_policy::ToolPolicy;

// Re-export from aion-types — single source of truth
pub use aion_types::spawner::{ForkOverrides, Spawner, SubAgentConfig, SubAgentResult};

/// Spawns independent child agents that share the parent's LLM provider.
///
/// Sub-agents use a [`NullSink`] so their streaming output is silently
/// discarded.  Results are collected via `engine.run()` and returned to the
/// parent which emits them as a single `tool_result` event — matching the
/// Claude Code pattern where only the parent writes to stdout.
///
/// Children inherit the parent's runtime tool policy. Fork overrides can
/// further narrow that policy, but cannot restore tools denied to the parent.
/// Host-supplied routing resolves only provider/model configuration, never tool permissions.
#[async_trait]
pub trait ChildRouter: Send + Sync {
    async fn resolve(&self, prompt: &str) -> Result<ChildProvider, String>;
}

pub struct ChildProvider {
    pub provider: Arc<dyn LlmProvider>,
    pub config: Config,
    /// Host-produced route metadata, returned alongside child output.
    pub decision: String,
}

pub struct AgentSpawner {
    provider: Arc<dyn LlmProvider>,
    child_router: Option<Arc<dyn ChildRouter>>,
    host_approval: Option<(Arc<ToolApprovalManager>, Arc<dyn ProtocolEmitter>)>,
    parent_plan: Option<Arc<AtomicBool>>,
    base_config: Config,
    cwd: PathBuf,
    runtime_env: Vec<(String, String)>,
    tool_policy: ToolPolicy,
}

impl AgentSpawner {
    pub fn new(provider: Arc<dyn LlmProvider>, config: Config, cwd: PathBuf, tool_policy: ToolPolicy) -> Self {
        Self::new_with_env(provider, config, cwd, Vec::new(), tool_policy)
    }

    pub fn new_with_env(
        provider: Arc<dyn LlmProvider>,
        config: Config,
        cwd: PathBuf,
        runtime_env: Vec<(String, String)>,
        tool_policy: ToolPolicy,
    ) -> Self {
        Self {
            provider,
            child_router: None,
            host_approval: None,
            parent_plan: None,
            base_config: config,
            cwd,
            runtime_env,
            tool_policy,
        }
    }

    pub fn with_child_router(mut self, router: Arc<dyn ChildRouter>) -> Self {
        self.child_router = Some(router);
        self
    }

    pub fn with_host_approval(mut self, manager: Arc<ToolApprovalManager>, writer: Arc<dyn ProtocolEmitter>) -> Self {
        self.host_approval = Some((manager, writer));
        self
    }
    pub fn with_parent_plan(mut self, flag: Arc<AtomicBool>) -> Self {
        self.parent_plan = Some(flag);
        self
    }

    fn configure_child_engine(&self, engine: &mut AgentEngine) {
        if self.child_router.is_some() {
            if let Some(flag) = &self.parent_plan {
                engine.set_plan_active_flag(flag.clone());
            }
            engine.pin_model();
            if let Some((manager, writer)) = &self.host_approval {
                engine.set_approval_manager(manager.clone());
                engine.set_protocol_writer(writer.clone());
                engine.set_approval_namespace(uuid::Uuid::now_v7().to_string());
            }
        }
    }

    async fn child_provider(
        &self,
        config: &mut Config,
        prompt: &str,
    ) -> Result<(Arc<dyn LlmProvider>, Option<String>), String> {
        let Some(router) = &self.child_router else {
            return Ok((self.provider.clone(), None));
        };
        if self.host_approval.is_none() {
            return Err("Hosted child routing requires the parent's approval channel".into());
        }
        // Live host mode, including yolo → default changes, owns approval.
        config.tools.auto_approve = false;
        let selected = router.resolve(prompt).await?;
        config.provider = selected.config.provider;
        config.provider_label = selected.config.provider_label;
        config.api_key = selected.config.api_key;
        config.base_url = selected.config.base_url;
        config.model = selected.config.model;
        config.compat = selected.config.compat;
        config.bedrock = selected.config.bedrock;
        config.vertex = selected.config.vertex;
        config.prompt_caching = selected.config.prompt_caching;
        Ok((selected.provider, Some(selected.decision)))
    }

    /// Spawn a single sub-agent and wait for result.
    pub async fn spawn_one(&self, sub_config: SubAgentConfig) -> SubAgentResult {
        let mut config = self.base_config.clone();
        config.max_turns = Some(sub_config.max_turns);
        config.max_tokens = Some(sub_config.max_tokens);
        if let Some(sp) = sub_config.system_prompt.clone() {
            config.system_prompt = Some(sp);
        }
        config.session.enabled = false;
        if self.child_router.is_none() {
            config.tools.auto_approve = true;
        }

        tracing::info!(target: "aion_agent", cwd = %self.cwd.display(), "sub-agent spawned with workspace cwd");

        let (provider, decision) = match self.child_provider(&mut config, &sub_config.prompt).await {
            Ok(provider) => provider,
            Err(error) => {
                return SubAgentResult {
                    name: sub_config.name,
                    text: format!("Child routing failed: {error}"),
                    usage: TokenUsage::default(),
                    turns: 0,
                    is_error: true,
                };
            }
        };
        let child_policy = effective_child_tool_policy(&self.tool_policy, &[]);
        let (tools, mcp) = self.build_child_tools(&config, &child_policy).await;
        let output: Arc<dyn OutputSink> = Arc::new(NullSink);
        let mut engine = AgentEngine::new_with_provider_and_env(
            provider,
            config,
            tools,
            output,
            self.cwd.clone(),
            self.runtime_env.clone(),
        );
        self.configure_child_engine(&mut engine);
        engine.set_tool_policy(child_policy);
        engine.set_mcp_catalog(mcp.catalog);

        match engine.run(&sub_config.prompt, "").await {
            Ok(result) => SubAgentResult {
                name: sub_config.name,
                text: match decision {
                    Some(decision) => format!("[Auto Router decision: {decision}]\n{}", result.text),
                    None => result.text,
                },
                usage: result.usage,
                turns: result.turns,
                is_error: false,
            },
            Err(e) => SubAgentResult {
                name: sub_config.name,
                text: match decision {
                    Some(decision) => format!("[Auto Router decision: {decision}]\nSub-agent error: {e}"),
                    None => format!("Sub-agent error: {e}"),
                },
                usage: TokenUsage::default(),
                turns: 0,
                is_error: true,
            },
        }
    }

    /// Spawn multiple sub-agents in parallel.
    pub async fn spawn_parallel(&self, sub_configs: Vec<SubAgentConfig>) -> Vec<SubAgentResult> {
        if self.child_router.is_none() {
            let futures: Vec<_> = sub_configs
                .into_iter()
                .map(|config| {
                    let spawner = self.clone_for_spawn();
                    tokio::spawn(async move { spawner.spawn_one(config).await })
                })
                .collect();
            let mut results = Vec::new();
            for future in futures {
                match future.await {
                    Ok(result) => results.push(result),
                    Err(e) => results.push(SubAgentResult {
                        name: "unknown".to_string(),
                        text: format!("Task join error: {e}"),
                        usage: TokenUsage::default(),
                        turns: 0,
                        is_error: true,
                    }),
                }
            }
            return results;
        }
        // Keep children in the caller's future: dropping a stopped parent must
        // cancel in-flight child inference/approval rather than detach tasks.
        futures::future::join_all(sub_configs.into_iter().map(|config| async move {
            std::panic::AssertUnwindSafe(self.spawn_one(config))
                .catch_unwind()
                .await
                .unwrap_or_else(|_| SubAgentResult {
                    name: "unknown".into(),
                    text: "Sub-agent task panicked".into(),
                    usage: TokenUsage::default(),
                    turns: 0,
                    is_error: true,
                })
        }))
        .await
    }

    fn clone_for_spawn(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            child_router: self.child_router.clone(),
            host_approval: self.host_approval.clone(),
            parent_plan: self.parent_plan.clone(),
            base_config: self.base_config.clone(),
            cwd: self.cwd.clone(),
            runtime_env: self.runtime_env.clone(),
            tool_policy: self.tool_policy.clone(),
        }
    }

    async fn build_child_tools(&self, config: &Config, policy: &ToolPolicy) -> (ToolRegistry, McpBootstrap) {
        let mut tools = build_tool_registry(policy, &self.cwd, &self.runtime_env);
        let mut builtin_names = tools.tool_names();
        builtin_names.push("ToolSearch".to_string());
        let bootstrap = AgentBootstrap::new(config.clone(), self.cwd.to_string_lossy(), Arc::new(NullSink))
            .runtime_env(self.runtime_env.clone());
        let mcp = bootstrap.connect_mcp(&mut tools, &builtin_names).await;
        if !config.mcp.servers.is_empty() {
            // MCP schemas are deferred by default, just as in the parent engine.
            // Respect the inherited tool policy in both discovery and execution.
            let definitions = tools.to_tool_defs_filtered(|tool| policy.allows(tool.name()));
            tools.register(Box::new(
                ToolSearchTool::new(definitions).with_documentation(tools.documentation()),
            ));
        }
        (tools, mcp)
    }
}

#[async_trait]
impl Spawner for AgentSpawner {
    async fn spawn_fork(&self, sub_config: SubAgentConfig, overrides: ForkOverrides) -> SubAgentResult {
        let mut config = self.base_config.clone();
        config.max_turns = Some(sub_config.max_turns);
        config.max_tokens = Some(sub_config.max_tokens);
        if let Some(sp) = sub_config.system_prompt.clone() {
            config.system_prompt = Some(sp);
        }
        config.session.enabled = false;
        if self.child_router.is_none() {
            config.tools.auto_approve = true;
        }
        if let Some(model) = overrides.model.clone() {
            config.model = model;
        }

        let (provider, decision) = match self.child_provider(&mut config, &sub_config.prompt).await {
            Ok(provider) => provider,
            Err(error) => {
                return SubAgentResult {
                    name: sub_config.name,
                    text: format!("Child routing failed: {error}"),
                    usage: TokenUsage::default(),
                    turns: 0,
                    is_error: true,
                };
            }
        };
        let child_policy = effective_child_tool_policy(&self.tool_policy, &overrides.allowed_tools);
        let (tools, mcp) = self.build_child_tools(&config, &child_policy).await;
        let output: Arc<dyn OutputSink> = Arc::new(NullSink);
        let mut engine = AgentEngine::new_with_provider_and_env(
            provider,
            config,
            tools,
            output,
            self.cwd.clone(),
            self.runtime_env.clone(),
        );
        engine.set_initial_reasoning_effort(overrides.effort.clone());
        self.configure_child_engine(&mut engine);
        engine.set_tool_policy(child_policy);
        engine.set_mcp_catalog(mcp.catalog);

        match engine.run(&sub_config.prompt, "").await {
            Ok(result) => SubAgentResult {
                name: sub_config.name,
                text: match decision {
                    Some(decision) => format!("[Auto Router decision: {decision}]\n{}", result.text),
                    None => result.text,
                },
                usage: result.usage,
                turns: result.turns,
                is_error: false,
            },
            Err(e) => SubAgentResult {
                name: sub_config.name,
                text: match decision {
                    Some(decision) => format!("[Auto Router decision: {decision}]\nSub-agent error: {e}"),
                    None => format!("Sub-agent error: {e}"),
                },
                usage: TokenUsage::default(),
                turns: 0,
                is_error: true,
            },
        }
    }
}

fn effective_child_tool_policy(parent: &ToolPolicy, allowed_tools: &[String]) -> ToolPolicy {
    if allowed_tools.is_empty() {
        return parent.clone();
    }

    ToolPolicy::allow_only(
        allowed_tools
            .iter()
            .filter(|tool_name| parent.allows(tool_name))
            .cloned(),
    )
}

fn build_tool_registry(policy: &ToolPolicy, cwd: &Path, runtime_env: &[(String, String)]) -> ToolRegistry {
    let all_tools: Vec<(&str, Box<dyn aion_tools::Tool>)> = vec![
        ("Read", Box::new(ReadTool::new(None))),
        ("Write", Box::new(WriteTool::new(None))),
        ("Edit", Box::new(EditTool::new(None))),
        (
            "ExecCommand",
            Box::new(ExecCommandTool::new_with_env(cwd.to_path_buf(), runtime_env.to_vec())),
        ),
        ("Grep", Box::new(GrepTool::new(cwd.to_path_buf()))),
        ("Glob", Box::new(GlobTool::new(cwd.to_path_buf()))),
    ];

    let mut registry = ToolRegistry::new();
    for (name, tool) in all_tools {
        if policy.allows(name) {
            registry.register(tool);
        }
    }
    registry
}

#[cfg(test)]
#[path = "spawner_test.rs"]
mod spawner_test;
