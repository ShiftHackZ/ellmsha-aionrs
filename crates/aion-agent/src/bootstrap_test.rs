use super::*;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use aion_config::config::{CliArgs, McpServerConfig, TransportType};
    use aion_protocol::events::ToolCategory;
    use aion_tools::Tool;
    use aion_types::tool::ToolResult;
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use crate::output::OutputSink;
    use crate::output::null_sink::NullSink;
    use crate::tool_policy::ToolPolicy;

    use super::*;

    struct DeferredTestTool(&'static str);

    #[async_trait]
    impl Tool for DeferredTestTool {
        fn name(&self) -> &str {
            self.0
        }

        fn description(&self) -> &str {
            "deferred test tool"
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }

        fn is_concurrency_safe(&self, _input: &Value) -> bool {
            true
        }

        async fn execute(&self, _input: Value) -> ToolResult {
            ToolResult {
                content: "unused".to_string(),
                is_error: false,
            }
        }

        fn category(&self) -> ToolCategory {
            ToolCategory::Info
        }

        fn is_deferred(&self) -> bool {
            true
        }
    }

    fn test_config() -> Config {
        // Owned defaults for tests; never import the developer's ambient config.
        let mut config = Config::resolve_isolated(&CliArgs {
            provider: Some("anthropic".to_string()),
            api_key: Some("sk-test".to_string()),
            base_url: None,
            model: Some("claude-sonnet-4-20250514".to_string()),
            max_tokens: Some(4096),
            thinking: None,
            thinking_budget: None,
            max_turns: None,
            max_tool_call_malformed_turns: None,
            max_tool_call_failure_turns: None,
            system_prompt: None,
            profile: None,
            auto_approve: false,
            project_dir: None,
        })
        .unwrap();
        config.isolated_session = false;
        config
    }

    #[test]
    fn host_delegation_suppresses_only_spawn_and_keeps_other_tools() {
        for host in [false, true] {
            let mut bootstrap = AgentBootstrap::new(test_config(), "/tmp", Arc::new(NullSink)).host_delegation(host);
            let provider = bootstrap.resolve_provider();
            let mut registry = bootstrap.build_builtin_registry(Path::new("/tmp"));
            bootstrap.register_agent_tools(&mut registry, &provider, Path::new("/tmp"), vec![], None);
            assert_eq!(registry.tool_names().iter().any(|n| n == "Spawn"), !host);
            assert!(registry.tool_names().iter().any(|n| n == "Read"));
        }
    }

    #[test]
    fn mcp_servers_with_runtime_env_uses_server_env_as_override() {
        let mut config = test_config();
        config.mcp.servers.insert(
            "stdio".to_string(),
            McpServerConfig {
                transport: TransportType::Stdio,
                command: Some("server".to_string()),
                args: None,
                env: Some(HashMap::from([
                    ("OVERRIDE".to_string(), "server".to_string()),
                    ("SERVER_ONLY".to_string(), "1".to_string()),
                ])),
                url: None,
                headers: None,
                deferred: None,
                startup_timeout_ms: None,
            },
        );

        let output: Arc<dyn OutputSink> = Arc::new(NullSink);
        let bootstrap = AgentBootstrap::new(config, "/tmp", output).runtime_env(vec![
            ("OVERRIDE".to_string(), "runtime".to_string()),
            ("RUNTIME_ONLY".to_string(), "1".to_string()),
        ]);

        let servers = bootstrap.mcp_servers_with_runtime_env();
        let env = servers
            .get("stdio")
            .and_then(|server| server.env.as_ref())
            .expect("stdio server env should exist");

        assert_eq!(env.get("OVERRIDE").map(String::as_str), Some("server"));
        assert_eq!(env.get("SERVER_ONLY").map(String::as_str), Some("1"));
        assert_eq!(env.get("RUNTIME_ONLY").map(String::as_str), Some("1"));
    }

    #[tokio::test]
    async fn tool_search_snapshot_excludes_policy_denied_tools() {
        let output: Arc<dyn OutputSink> = Arc::new(NullSink);
        let bootstrap = AgentBootstrap::new(test_config(), "/tmp", output)
            .tool_policy(ToolPolicy::allow_only(["ToolSearch", "AllowedDeferred"]));
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DeferredTestTool("AllowedDeferred")));
        registry.register(Box::new(DeferredTestTool("DeniedDeferred")));

        bootstrap.register_tool_search(&mut registry);

        let tool_search = registry.get("ToolSearch").expect("ToolSearch should be registered");
        let allowed = tool_search.execute(json!({"query": "AllowedDeferred"})).await;
        let denied = tool_search.execute(json!({"query": "DeniedDeferred"})).await;

        assert!(allowed.content.contains("AllowedDeferred"));
        assert!(denied.content.starts_with("No deferred tools matching"));
        assert!(!denied.content.contains("\"name\": \"DeniedDeferred\""));
    }

    #[tokio::test]
    async fn isolated_bootstrap_omits_ambient_instructions_memory_and_all_skills() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("AGENTS.md"), "AMBIENT_PROJECT_MARKER").unwrap();
        let skill_root = workspace.path().join(".aionrs/skills/ambient-skill");
        std::fs::create_dir_all(&skill_root).unwrap();
        std::fs::write(
            skill_root.join("SKILL.md"),
            "---\nname: ambient-skill\ndescription: Ambient fixture\n---\nAMBIENT_SKILL_MARKER",
        )
        .unwrap();
        let mut config = test_config();
        config.isolated_session = true;
        config.system_prompt = Some("EXPLICIT_SESSION_MARKER".into());
        let mut bootstrap = AgentBootstrap::new(config, workspace.path().to_string_lossy(), Arc::new(NullSink))
            .extra_skill_dirs(vec![skill_root]);
        let environment = bootstrap.resolve_environment(workspace.path().into()).unwrap();
        assert!(environment.memory_dir.is_none());
        let skills = bootstrap.load_skills(workspace.path(), None).await;
        assert!(skills.is_empty(), "even bundled and explicit extra skills are omitted");
        bootstrap.configure_system_prompt(&environment, &skills);
        let prompt = bootstrap.config.system_prompt.as_ref().unwrap();
        assert!(prompt.contains("EXPLICIT_SESSION_MARKER"));
        assert!(!prompt.contains("AMBIENT_PROJECT_MARKER"));
        assert!(!prompt.contains("AMBIENT_SKILL_MARKER"));
        assert!(!prompt.contains("auto memory"));
        let provider = bootstrap.resolve_provider();
        let mut registry = ToolRegistry::new();
        bootstrap.register_agent_tools(&mut registry, &provider, workspace.path(), skills, None);
        assert!(registry.get("Skill").is_none());
        assert!(registry.get("Spawn").is_some());
    }
}
