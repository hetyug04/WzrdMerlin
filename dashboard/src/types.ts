export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  isThinking?: boolean;
}

export interface ReasoningBlock {
  id: string;
  type: "thinking" | "narrative" | "action" | "observation";
  content: string;
  /** Full-fidelity payload shown in the expanded view (e.g. pretty-printed args JSON). */
  detail?: string;
  timestamp: string;
  duration?: number;
  status?: "success" | "pending";
}

export interface MerlinEvent {
  id: string;
  type: string;
  timestamp: string;
  source_actor: string;
  target_actor?: string;
  correlation_id: string;
  payload: any;
}

export interface SystemInfo {
  total_memory: number;
  used_memory: number;
  cpu_usage: number;
}

export interface TelemetryState {
  cpu: number;
  ram: number;
  vram: number;
  vramTotal: number;
  temperature: number;
  tokensPerSec: number;
  latencyMs: number;
  history: { time: number; cpu: number; ram: number }[];
}

export interface Actor {
  id: string;
  name: string;
  status: "ACTIVE" | "BUSY" | "IDLE" | "SLEEP";
  currentTask?: string;
  meshEvent?: string;
}

export interface Tool {
  name: string;
  type: "base" | "synthesized";
  version?: string;
  isNew?: boolean;
}
