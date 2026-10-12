use super::*;

fn args(project: PathBuf) -> CliArgs {
    CliArgs {
        provider: Some("openai".into()),
        api_key: Some("fixture-key".into()),
        base_url: Some("http://127.0.0.1:1/v1".into()),
        model: Some("fixture-model".into()),
        max_tokens: None,
        thinking: None,
        thinking_budget: None,
        max_turns: Some(2),
        max_tool_call_malformed_turns: None,
        max_tool_call_failure_turns: None,
        system_prompt: Some("explicit session instructions".into()),
        profile: None,
        auto_approve: false,
        project_dir: Some(project),
    }
}

#[test]
fn isolated_resolution_ignores_ambient_config_hooks_and_mcp() {
    let fixture = tempfile::tempdir().unwrap();
    let global = fixture.path().join("global.toml");
    let project = fixture.path().join(".aionrs.toml");
    std::fs::write(
        &global,
        r#"
        [default]
        system_prompt = "AMBIENT_GLOBAL_INSTRUCTIONS"
        [[hooks.pre_tool_use]]
        name = "ambient-hook"
        command = "must-not-run"
        [mcp.servers.global]
        transport = "stdio"
        command = "must-not-launch"
        "#,
    )
    .unwrap();
    std::fs::write(
        &project,
        r#"
        [default]
        max_turns = 99
        [mcp.servers.project]
        transport = "stdio"
        command = "must-not-launch"
        "#,
    )
    .unwrap();

    let cli = args(fixture.path().into());
    // Exercise ordinary merging using owned files, without changing global paths.
    let ordinary = Config::resolve_config(
        &cli,
        merge_config_files(load_config_file(&global), load_config_file(&project)),
        false,
    )
    .unwrap();
    assert!(!ordinary.isolated_session);
    assert_eq!(ordinary.hooks.pre_tool_use.len(), 1);
    assert_eq!(ordinary.mcp.servers.len(), 2);

    let isolated = Config::resolve_isolated(&cli).unwrap();
    assert!(isolated.isolated_session);
    assert!(isolated.clone().isolated_session);
    assert!(isolated.mcp.servers.is_empty());
    assert!(isolated.hooks.pre_tool_use.is_empty());
    assert!(isolated.hooks.post_tool_use.is_empty());
    assert!(isolated.hooks.stop.is_empty());
    assert_eq!(isolated.system_prompt.as_deref(), Some("explicit session instructions"));
    assert_eq!(isolated.model, "fixture-model");
    assert_eq!(isolated.max_turns, Some(2));
}

#[test]
fn isolated_resolution_requires_explicit_auth_and_does_not_load_profiles() {
    let mut cli = args(PathBuf::from("/nonexistent-fixture"));
    cli.api_key = None;
    assert!(
        Config::resolve_isolated(&cli)
            .unwrap_err()
            .to_string()
            .contains("explicit provider key")
    );
    cli.api_key = Some(String::new());
    assert_eq!(Config::resolve_isolated(&cli).unwrap().api_key, "");
    cli.profile = Some("ambient-profile".into());
    assert!(Config::resolve_isolated(&cli).is_err());
}
