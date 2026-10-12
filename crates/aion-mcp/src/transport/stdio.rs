use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use super::{McpError, McpTransport};
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

/// Stdio transport: communicates with MCP server via child process stdin/stdout
pub struct StdioTransport {
    stdin: Mutex<BufWriter<ChildStdin>>,
    stdout: Mutex<Lines<BufReader<ChildStdout>>>,
    child: Mutex<Child>,
    next_id: AtomicU64,
    request_lock: Mutex<()>,
    tools_changed: AtomicBool,
}

impl StdioTransport {
    /// Spawn a child process and return the transport
    pub async fn spawn(command: &str, args: &[String], env: &HashMap<String, String>) -> Result<Self, McpError> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.kill_on_drop(true)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .envs(env);

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("Failed to spawn '{}': {}", command, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture child stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture child stdout".into()))?;

        Ok(Self {
            stdin: Mutex::new(BufWriter::new(stdin)),
            stdout: Mutex::new(BufReader::new(stdout).lines()),
            child: Mutex::new(child),
            next_id: AtomicU64::new(1),
            request_lock: Mutex::new(()),
            tools_changed: AtomicBool::new(false),
        })
    }

    /// Get the next request ID
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Send a JSON-RPC message (request or notification) via stdin
    async fn send(&self, req: &JsonRpcRequest) -> Result<(), McpError> {
        let json =
            serde_json::to_string(req).map_err(|e| McpError::Transport(format!("JSON serialize error: {}", e)))?;

        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(json.as_bytes())
            .await
            .map_err(|e| McpError::Transport(format!("Write to stdin failed: {}", e)))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::Transport(format!("Write newline failed: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| McpError::Transport(format!("Flush stdin failed: {}", e)))?;
        Ok(())
    }

    /// Read a single JSON-RPC response from stdout
    async fn read_response(&self, expected_id: u64) -> Result<JsonRpcResponse, McpError> {
        let mut stdout = self.stdout.lock().await;
        loop {
            // Lines::next_line is cancellation-safe when a bounded catalog refresh times out.
            let line = stdout
                .next_line()
                .await
                .map_err(|error| McpError::Transport(format!("Read from stdout failed: {error}")))?
                .ok_or_else(|| McpError::Transport("Child process stdout closed".into()))?;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)
                .map_err(|error| McpError::Transport(format!("Invalid JSON-RPC response: {error}")))?;
            if value.get("method").is_some() {
                if value["method"] == "notifications/tools/list_changed" {
                    self.tools_changed.store(true, Ordering::Release);
                }
                continue;
            }
            let response: JsonRpcResponse = serde_json::from_value(value)
                .map_err(|error| McpError::Transport(format!("Invalid JSON-RPC response: {error}")))?;
            // Discard late responses to timed-out requests, never treat them as the next call's result.
            if response.id == Some(expected_id) {
                return Ok(response);
            }
        }
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(&self, req: &JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        let _request = self.request_lock.lock().await;
        let request = JsonRpcRequest::new(self.next_id(), &req.method, req.params.clone());
        self.send(&request).await?;
        let response = self
            .read_response(request.id.expect("request always has an ID"))
            .await?;

        // Check for JSON-RPC error in response
        if let Some(err) = &response.error {
            return Err(McpError::JsonRpc {
                code: err.code,
                message: err.message.clone(),
            });
        }

        Ok(response)
    }

    async fn notify(&self, req: &JsonRpcRequest) -> Result<(), McpError> {
        self.send(req).await
    }

    fn take_tools_changed(&self) -> bool {
        self.tools_changed.swap(false, Ordering::AcqRel)
    }

    async fn close(&self) -> Result<(), McpError> {
        // Drop stdin to signal EOF, then wait for child
        let mut child = self.child.lock().await;
        // kill the child process gracefully
        let _ = child.kill().await;
        Ok(())
    }
}

#[cfg(test)]
#[path = "stdio_test.rs"]
mod stdio_test;
