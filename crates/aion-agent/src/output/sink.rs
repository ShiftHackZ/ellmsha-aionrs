/// Abstraction over output channels (terminal vs JSON stream protocol)
pub trait OutputSink: Send + Sync {
    /// Stream text delta from LLM
    fn emit_text_delta(&self, text: &str, msg_id: &str);

    /// Stream thinking content from LLM
    fn emit_thinking(&self, text: &str, msg_id: &str);

    /// Announce a tool call.
    fn emit_tool_call(&self, tool_use_id: &str, name: &str, input: &str);

    /// Display tool result.
    fn emit_tool_result(&self, tool_use_id: &str, name: &str, is_error: bool, content: &str);

    /// Signal start of a new message stream
    fn emit_stream_start(&self, msg_id: &str);

    /// Signal end of a message stream with usage stats
    fn emit_stream_end(
        &self,
        msg_id: &str,
        turns: usize,
        input_tokens: u64,
        output_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
    );

    /// Report one completed provider request without exposing prompt content.
    fn emit_usage_report(&self, _report: serde_json::Value) {}

    /// Billable internal work must not overwrite the visible chat's context.
    fn emit_auxiliary_usage(&self, model: &str, usage: &aion_types::message::TokenUsage) {
        if usage.input_tokens == 0 && usage.output_tokens == 0 {
            return;
        }
        self.emit_usage_report(serde_json::json!({
            "request_id":uuid::Uuid::now_v7().to_string(),"model":model,
            "input_tokens":usage.input_tokens,"output_tokens":usage.output_tokens,
            "cache_read_tokens":usage.cache_read_tokens,"cache_creation_tokens":usage.cache_creation_tokens,
            "context_tokens":0,"context_window":0,"context_reported":false,"breakdown":{}
        }));
    }

    /// Display error
    fn emit_error(&self, msg: &str);

    /// Display informational message
    fn emit_info(&self, msg: &str);
}
