use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tool_defs() -> Vec<ToolDef> {
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
