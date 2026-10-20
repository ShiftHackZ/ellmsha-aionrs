use super::OutputSink;

/// Silent output sink that discards all output.
///
/// Used for sub-agents whose results are collected via `engine.run()` return
/// value and emitted by the parent as a single `tool_result` event.  This
/// prevents raw text from leaking into the parent's protocol stream (JSON
/// Lines) — aligning with the Claude Code pattern where sub-agents never
/// write directly to stdout.
pub struct NullSink;

impl OutputSink for NullSink {
    fn emit_text_delta(&self, _text: &str, _msg_id: &str) {}
    fn emit_thinking(&self, _text: &str, _msg_id: &str) {}
    fn emit_tool_call(&self, _tool_use_id: &str, _name: &str, _input: &str) {}
    fn emit_tool_result(&self, _tool_use_id: &str, _name: &str, _is_error: bool, _content: &str) {}
    fn emit_stream_start(&self, _msg_id: &str) {}
    fn emit_stream_end(
        &self,
        _msg_id: &str,
        _turns: usize,
        _input_tokens: u64,
        _output_tokens: u64,
        _cache_creation_tokens: u64,
        _cache_read_tokens: u64,
    ) {
    }
    fn emit_error(&self, _msg: &str) {}
    fn emit_info(&self, _msg: &str) {}
}

#[cfg(test)]
#[path = "null_sink_test.rs"]
mod null_sink_test;

/// Keep child text silent while forwarding accounting to the owning conversation.
pub struct UsageOnlySink {
    pub parent: std::sync::Arc<dyn OutputSink>,
    pub provider_id: Option<String>,
}
impl OutputSink for UsageOnlySink {
    fn emit_text_delta(&self, _: &str, _: &str) {}
    fn emit_thinking(&self, _: &str, _: &str) {}
    fn emit_tool_call(&self, _: &str, _: &str, _: &str) {}
    fn emit_tool_result(&self, _: &str, _: &str, _: bool, _: &str) {}
    fn emit_stream_start(&self, _: &str) {}
    fn emit_stream_end(&self, _: &str, _: usize, _: u64, _: u64, _: u64, _: u64) {}
    fn emit_error(&self, _: &str) {}
    fn emit_info(&self, _: &str) {}
    fn emit_usage_report(&self, mut report: serde_json::Value) {
        report["is_child"] = serde_json::json!(true);
        if let Some(provider) = &self.provider_id {
            report["provider_id"] = serde_json::json!(provider);
        }
        self.parent.emit_usage_report(report);
    }
}
