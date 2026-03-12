use sysinfo::System;
use std::env;
use std::time::Duration;
use tokio::time::sleep;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::Utc;

#[derive(Serialize, Deserialize, Debug)]
struct Event {
    pub id: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub timestamp: String,
    pub source_actor: String,
    pub correlation_id: String,
    pub payload: serde_json::Value,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    println!("Sidecar: Connecting to NATS at {}", nats_url);
    
    // Retry loop for NATS connection
    let client = loop {
        match async_nats::connect(&nats_url).await {
            Ok(c) => break c,
            Err(e) => {
                println!("Sidecar: Connection to NATS failed: {}. Retrying in 5s...", e);
                sleep(Duration::from_secs(5)).await;
            }
        }
    };
    
    let mut sys = System::new_all();
    
    println!("Sidecar: Starting hardware telemetry loop...");
    
    loop {
        sys.refresh_all();
        
        let total_memory = sys.total_memory();
        let used_memory = sys.used_memory();
        let cpu_usage = sys.global_cpu_info().cpu_usage();
        
        let payload = serde_json::json!({
            "ram_usage": (used_memory as f64 / total_memory as f64) * 100.0,
            "cpu_usage": cpu_usage,
            "total_memory": total_memory,
            "used_memory": used_memory,
            "timestamp": Utc::now().to_rfc3339()
        });
        
        let event = Event {
            id: Uuid::new_v4().to_string(),
            event_type: "system.heartbeat".to_string(),
            timestamp: Utc::now().to_rfc3339(),
            source_actor: "rust-sidecar".to_string(),
            correlation_id: "system".to_string(),
            payload,
        };
        
        let message = serde_json::to_vec(&event)?;
        if let Err(e) = client.publish("events.system.heartbeat", message.into()).await {
            println!("Sidecar: Failed to publish heartbeat: {}", e);
        }
        
        sleep(Duration::from_secs(2)).await;
    }
}
