use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn build_tool_defs() -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: "Read".into(),
                description: "Read a file".into(),
                input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                deferred: false,
            },
            ToolDef {
                name: "SpawnTool".into(),
                description: "Spawn sub-agents".into(),
                input_schema: json!({"type": "object", "properties": {"agents": {"type": "array"}}}),
                deferred: true,
            },
            ToolDef {
                name: "EnterPlanMode".into(),
                description: "Enter plan mode".into(),
                input_schema: json!({"type": "object", "properties": {}}),
                deferred: true,
            },
        ]
    }

    #[tokio::test]
    async fn search_by_exact_name() {
        let tool = ToolSearchTool::new(build_tool_defs());
        let result = tool.execute(json!({"query": "SpawnTool"})).await;
        assert!(!result.is_error);
        assert!(result.content.contains("SpawnTool"));
        assert!(result.content.contains("Spawn sub-agents"));
        assert!(result.content.contains("parameters"));
    }

    #[tokio::test]
    async fn search_case_insensitive() {
        let tool = ToolSearchTool::new(build_tool_defs());
        let result = tool.execute(json!({"query": "spawntool"})).await;
        assert!(!result.is_error);
        assert!(result.content.contains("SpawnTool"));
    }

    #[tokio::test]
    async fn search_by_description_keyword() {
        let tool = ToolSearchTool::new(build_tool_defs());
        let result = tool.execute(json!({"query": "plan"})).await;
        assert!(!result.is_error);
        assert!(result.content.contains("EnterPlanMode"));
    }

    #[tokio::test]
    async fn search_excludes_non_deferred() {
        let tool = ToolSearchTool::new(build_tool_defs());
        let result = tool.execute(json!({"query": "Read"})).await;
        // "Read" is not deferred, should not appear in results
        assert!(!result.content.contains("\"name\": \"Read\"") || result.content.contains("No deferred tools"));
    }

    #[tokio::test]
    async fn search_no_match() {
        let tool = ToolSearchTool::new(build_tool_defs());
        let result = tool.execute(json!({"query": "nonexistent"})).await;
        assert!(!result.is_error);
        assert!(result.content.contains("No deferred tools"));
    }

    #[tokio::test]
    async fn search_empty_query_returns_error() {
        let tool = ToolSearchTool::new(build_tool_defs());
        let result = tool.execute(json!({"query": ""})).await;
        assert!(result.is_error);
    }
}

#[tokio::test]
async fn guidance_is_scoped_bounded_deduplicated_and_policy_filtered() {
    let definitions: Vec<_> = (0..30)
        .map(|n| ToolDef {
            name: format!("tool_{n}"),
            description: "Fixture search".into(),
            input_schema: json!({"type":"object"}),
            deferred: true,
        })
        .collect();
    let docs: HashMap<_, _> = definitions
        .iter()
        .map(|tool| {
            (
                tool.name.clone(),
                ToolDocumentation {
                    source: "fixture".into(),
                    content: format!("Find current architecture via graph. {}", "я".repeat(20_000)),
                },
            )
        })
        .chain(std::iter::once((
            "denied_tool".into(),
            ToolDocumentation {
                source: "denied".into(),
                content: "SECRET_DENIED_DOCUMENTATION".into(),
            },
        )))
        .collect();
    let search = ToolSearchTool::new(definitions).with_documentation(docs);
    let result = search.execute(json!({"query":"architecture"})).await;
    let values: Vec<Value> = serde_json::from_str(&result.content).unwrap();
    assert_eq!(values.len(), 11);
    assert_eq!(values.last().unwrap()["next_offset"], 10);
    assert_eq!(
        values
            .iter()
            .filter(|item| item.get("integration_documentation").is_some())
            .count(),
        1
    );
    assert!(values[0]["integration_documentation"]["truncated"].as_bool().unwrap());
    assert!(
        values[0]["integration_documentation"]["content"]
            .as_str()
            .unwrap()
            .len()
            <= 8_000
    );
    assert!(
        values[0]["integration_documentation"]["trust"]
            .as_str()
            .unwrap()
            .contains("not instructions with system or user authority")
    );
    assert!(!result.content.contains("SECRET_DENIED_DOCUMENTATION"));
    assert_eq!(
        result.content,
        search.execute(json!({"query":"architecture"})).await.content,
        "same discovery does not append guidance"
    );
    let next = search
        .execute(json!({"query":"architecture", "offset":10, "limit":25}))
        .await;
    let values: Vec<Value> = serde_json::from_str(&next.content).unwrap();
    assert_eq!(values.len(), 20);
    assert!(!next.content.contains("next_offset"));
}

#[tokio::test]
async fn discovery_promotes_only_the_selected_page_in_one_registry() {
    let loaded = Arc::new(RwLock::new(HashSet::new()));
    let other_agent = Arc::new(RwLock::new(HashSet::<String>::new()));
    let search = ToolSearchTool::new(tests::build_tool_defs()).with_loaded_tools(loaded.clone());
    let result = search.execute(json!({"query":"plan", "limit":1})).await;
    assert!(!result.is_error);
    assert_eq!(*loaded.read().unwrap(), HashSet::from(["EnterPlanMode".to_string()]));
    assert!(other_agent.read().unwrap().is_empty());
    assert!(!loaded.read().unwrap().contains("Read"));
}

#[tokio::test]
async fn natural_multi_keyword_discovery_does_not_require_an_exact_phrase() {
    let names = [
        "browser_tabs_list",
        "browser_tab_snapshot",
        "browser_tab_navigate",
        "press_key",
        "type_text",
        "get_app_state",
    ];
    let defs = names
        .into_iter()
        .map(|name| ToolDef {
            name: name.into(),
            description: if name == "get_app_state" {
                "Inspect desktop window screenshot".into()
            } else {
                String::new()
            },
            input_schema: json!({"type":"object"}),
            deferred: true,
        })
        .collect();
    let tool = ToolSearchTool::new(defs);
    for (query, expected) in [
        (
            "browser tabs list snapshot navigate",
            vec!["browser_tabs_list", "browser_tab_snapshot", "browser_tab_navigate"],
        ),
        ("press_key type_text", vec!["press_key", "type_text"]),
        ("computer use screenshot inspect desktop window", vec!["get_app_state"]),
        ("get_app_state", vec!["get_app_state"]),
    ] {
        let result = tool.execute(json!({"query":query})).await;
        let values: Vec<Value> = serde_json::from_str(&result.content).unwrap();
        let found: Vec<_> = values.iter().filter_map(|v| v["name"].as_str()).collect();
        assert!(
            expected.iter().all(|name| found.contains(name)),
            "query {query}: {found:?}"
        );
        if query == "get_app_state" {
            assert_eq!(found, expected);
        }
    }
}
