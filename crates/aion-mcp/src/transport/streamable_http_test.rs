use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Servers built on the MCP Python SDK / `fastmcp` use `sse-starlette`,
    /// whose default SSE line separator is CRLF (`\r\n`). The event terminator
    /// is therefore `\r\n\r\n`, which does not contain `\n\n`. The parser must
    /// still recover the JSON-RPC response, otherwise such servers connect but
    /// expose no tools to the model.
    #[test]
    fn extracts_jsonrpc_from_crlf_delimited_sse() {
        let body = "event: message\r\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[]}}\r\n\r\n";
        let resp = extract_jsonrpc_from_sse_buffer(body).expect("should parse CRLF SSE");
        assert_eq!(resp.id, Some(2));
        assert!(resp.result.is_some());
    }

    /// Node `@modelcontextprotocol/sdk` servers emit LF-delimited SSE.
    #[test]
    fn extracts_jsonrpc_from_lf_delimited_sse() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n";
        let resp = extract_jsonrpc_from_sse_buffer(body).expect("should parse LF SSE");
        assert_eq!(resp.id, Some(1));
    }

    /// Data split across multiple `data:` lines must be reassembled.
    #[test]
    fn reassembles_multiline_data() {
        let body = "data: {\"jsonrpc\":\"2.0\",\r\ndata: \"id\":7,\"result\":{}}\r\n\r\n";
        let resp = extract_jsonrpc_from_sse_buffer(body).expect("should join data lines");
        assert_eq!(resp.id, Some(7));
    }

    /// Notifications (no `id`) and comment/ping lines must be skipped.
    #[test]
    fn returns_none_without_complete_response() {
        let body = ": keep-alive\r\n\r\nevent: message\r\ndata: not-json\r\n\r\n";
        assert!(extract_jsonrpc_from_sse_buffer(body).is_none());
    }
}

#[test]
fn notifications_do_not_replace_the_rpc_response() {
    let stream = concat!(
        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n",
        "data: {\"jsonrpc\":\"2.0\",\"id\":42,\"result\":{\"tools\":[]}}\n\n"
    );
    let response = extract_jsonrpc_from_sse_buffer(stream).unwrap();
    assert_eq!(response.id, Some(42));
    assert_eq!(response.result.unwrap()["tools"], serde_json::json!([]));
}
