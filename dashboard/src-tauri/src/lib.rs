use sysinfo::System;

#[tauri::command]
fn get_system_info() -> serde_json::Value {
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let total_memory = sys.total_memory();
    let used_memory = sys.used_memory();
    
    serde_json::json!({
        "total_memory": total_memory,
        "used_memory": used_memory,
         "cpu_usage": sys.global_cpu_info().cpu_usage(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_log::Builder::new().build())
    .invoke_handler(tauri::generate_handler![get_system_info])
    .setup(|app| {
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
