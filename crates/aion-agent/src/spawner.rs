use aion_protocol::{ToolApprovalManager, writer::ProtocolEmitter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use async_trait::async_trait;
use futures::FutureExt;

use crate::skill_tool::SkillTool;
use aion_config::config::Config;
use aion_providers::LlmProvider;
use aion_skills::loader::load_all_skills;
use aion_skills::permissions::SkillPermissionChecker;
use aion_skills::prompt::format_skills_within_budget;
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
use crate::output::null_sink::{NullSink, UsageOnlySink};
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

/// Host-owned barrier run before a child's routing, tools or model execution.
#[async_trait]
pub trait ChildPromptPreflight: Send + Sync {
    async fn instructions(&self) -> Result<String, String>;
}

pub struct AgentSpawner {
    provider: Arc<dyn LlmProvider>,
    child_router: Option<Arc<dyn ChildRouter>>,
    child_preflight: Option<Arc<dyn ChildPromptPreflight>>,
    host_approval: Option<(Arc<ToolApprovalManager>, Arc<dyn ProtocolEmitter>)>,
    parent_plan: Option<Arc<AtomicBool>>,
    usage_output: Option<Arc<dyn OutputSink>>,
    base_config: Config,
    cwd: PathBuf,
    runtime_env: Vec<(String, String)>,
    tool_policy: ToolPolicy,
    skill_dirs: Vec<PathBuf>,
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
            child_preflight: None,
            host_approval: None,
            parent_plan: None,
            usage_output: None,
            base_config: config,
            cwd,
            runtime_env,
            tool_policy,
            skill_dirs: Vec::new(),
        }
    }

    pub(crate) fn with_skill_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.skill_dirs = dirs;
        self
    }

    pub fn with_usage_output(mut self, output: Arc<dyn OutputSink>) -> Self {
        self.usage_output = Some(output);
        self
    }

    fn child_output(&self, decision: &Option<String>) -> Arc<dyn OutputSink> {
        match &self.usage_output {
            Some(parent) => {
                let provider_id = decision
                    .as_ref()
                    .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
                    .and_then(|d| d["target"]["provider_id"].as_str().map(str::to_owned));
                Arc::new(UsageOnlySink {
                    parent: parent.clone(),
                    provider_id,
                })
            }
            None => Arc::new(NullSink),
        }
    }

    pub fn with_child_preflight(mut self, preflight: Arc<dyn ChildPromptPreflight>) -> Self {
        self.child_preflight = Some(preflight);
        self
    }

    async fn prepare_child_prompt(&self, sub: &SubAgentConfig, config: &mut Config) -> Result<String, SubAgentResult> {
        let Some(preflight) = &self.child_preflight else {
            return Ok(sub.prompt.clone());
        };
        let rules = preflight.instructions().await.map_err(|error| SubAgentResult {
            name: sub.name.clone(),
            text: error,
            usage: TokenUsage::default(),
            turns: 0,
            is_error: true,
        })?;
        if rules.is_empty() {
            return Ok(sub.prompt.clone());
        }
        config.system_prompt = Some(format!(
            "{rules}\n\n{}",
            config.system_prompt.take().unwrap_or_default()
        ));
        Ok(format!("{rules}\n\n{}", sub.prompt))
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
        if let Some(flag) = &self.parent_plan {
            engine.set_plan_active_flag(flag.clone());
        }
        if self.child_router.is_some() {
            engine.pin_model();
        }
        if let Some((manager, writer)) = &self.host_approval {
            engine.set_approval_manager(manager.clone());
            engine.set_protocol_writer(writer.clone());
            engine.set_approval_namespace(uuid::Uuid::now_v7().to_string());
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
        let prompt = match self.prepare_child_prompt(&sub_config, &mut config).await {
            Ok(prompt) => prompt,
            Err(result) => return result,
        };
        config.session.enabled = false;
        if self.host_approval.is_some() {
            // Live host mode owns approvals, including ordinary unrouted children.
            config.tools.auto_approve = false;
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
        let (tools, mcp) = self.build_child_tools(&mut config, &child_policy).await;
        let output = self.child_output(&decision);
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

        match engine.run(&prompt, "").await {
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

    async fn build_child_tools(&self, config: &mut Config, policy: &ToolPolicy) -> (ToolRegistry, McpBootstrap) {
        let mut tools = build_tool_registry(policy, &self.cwd, &self.runtime_env);
        let mut builtin_names = tools.tool_names();
        builtin_names.push("ToolSearch".to_string());
        let bootstrap = AgentBootstrap::new(config.clone(), self.cwd.to_string_lossy(), Arc::new(NullSink))
            .runtime_env(self.runtime_env.clone());
        let mcp = bootstrap.connect_mcp(&mut tools, &builtin_names).await;
        if !config.isolated_session && policy.allows("Skill") {
            let skills = load_all_skills(&self.cwd, &self.skill_dirs, false, mcp.manager.as_deref()).await;
            let visible: Vec<_> = skills
                .iter()
                .filter(|skill| !skill.disable_model_invocation)
                .cloned()
                .collect();
            // The parent's prepared prompt already carries its bounded catalog.
            // An explicit fork prompt still needs the same discoverability.
            if !config
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("The following skills are available for use with the Skill tool:")
            {
                let listing = format_skills_within_budget(&visible, None);
                if !listing.is_empty() {
                    config.system_prompt = Some(format!(
                        "{}\n\nThe following skills are available for use with the Skill tool:\n{listing}",
                        config.system_prompt.as_deref().unwrap_or_default()
                    ));
                }
            }
            let checker = SkillPermissionChecker::new(
                config.tools.skills.deny.clone(),
                config.tools.skills.allow.clone(),
                config.tools.auto_approve,
            );
            tools.register(Box::new(SkillTool::new(Arc::new(skills), self.cwd.clone(), checker)));
        }
        if policy.allows("ToolSearch") {
            // MCP schemas are deferred by default, just as in the parent engine.
            // Respect the inherited tool policy in both discovery and execution.
            let definitions = tools.to_tool_defs_filtered(|tool| policy.allows(tool.name()));
            tools.register(Box::new(
                ToolSearchTool::new(definitions)
                    .with_documentation(tools.documentation())
                    .with_loaded_tools(tools.loaded_tools()),
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
        let prompt = match self.prepare_child_prompt(&sub_config, &mut config).await {
            Ok(prompt) => prompt,
            Err(result) => return result,
        };
        config.session.enabled = false;
        if self.host_approval.is_some() {
            // Live host mode owns approvals, including ordinary unrouted children.
            config.tools.auto_approve = false;
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
        let (tools, mcp) = self.build_child_tools(&mut config, &child_policy).await;
        let output = self.child_output(&decision);
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

        match engine.run(&prompt, "").await {
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
