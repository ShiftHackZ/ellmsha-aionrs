use super::*;

// Isolated subprocess fixture: no real MCP process, workspace or user state is touched.
#[cfg(unix)]
#[tokio::test]
async fn notification_and_late_response_do_not_corrupt_the_next_call() {
    let script = r#"
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request.get('id') is None: continue
    if request['method'] == 'slow': time.sleep(0.08)
    print(json.dumps({'jsonrpc':'2.0','method':'notifications/tools/list_changed'}), flush=True)
    print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':{'method':request['method']}}), flush=True)
"#;
    let transport = StdioTransport::spawn("python3", &["-u".into(), "-c".into(), script.into()], &HashMap::new())
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            transport.request(&JsonRpcRequest::new(1, "slow", None))
        )
        .await
        .is_err()
    );
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        transport.request(&JsonRpcRequest::new(1, "fast", None)),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.result.unwrap()["method"], "fast");
    assert!(transport.take_tools_changed());
    assert!(!transport.take_tools_changed());
    transport.close().await.unwrap();
}
