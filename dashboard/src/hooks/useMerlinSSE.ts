import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  Message,
  ReasoningBlock,
  MerlinEvent,
  TelemetryState,
  SystemInfo,
} from "../types";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: any;
  }
}

/** Render a tool call as a readable one-liner for the ACTION trace block. */
function formatToolCall(tool: string, args?: Record<string, any>): string {
  if (!tool) return "tool()";
  if (!args || Object.keys(args).length === 0) return `${tool}()`;

  // For well-known tools, surface the most useful arg inline
  if (tool === "done" && args.summary) return `done("${args.summary}")`;
  if (tool === "request_human" && args.question)
    return `request_human("${args.question}")`;
  if (tool === "python_repl" && args.code)
    return `python_repl(\`${args.code.slice(0, 80)}${args.code.length > 80 ? "…" : ""}\`)`;
  if (tool === "shell_exec" && args.command)
    return `shell_exec("${args.command.slice(0, 80)}")`;
  if (tool === "search" && args.query)
    return `search("${args.query}")`;
  if (tool === "file_read" && args.path)
    return `file_read("${args.path}")`;
  if (tool === "file_write" && args.path)
    return `file_write("${args.path}")`;

  // Generic fallback: show first arg value
  const firstVal = Object.values(args)[0];
  const preview =
    typeof firstVal === "string"
      ? `"${firstVal.slice(0, 60)}${firstVal.length > 60 ? "…" : ""}"`
      : JSON.stringify(firstVal).slice(0, 60);
  return `${tool}(${preview})`;
}

export function useMerlinSSE() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "System initialized. DisCo agent is active and connected to the orchestration layer. How can I assist you today?",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    },
  ]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingStart, setThinkingStart] = useState<number | null>(null);
  const [liveThinkingText, setLiveThinkingText] = useState("");
  const [isWaitingForHuman, setIsWaitingForHuman] = useState(false);
  const [traces, setTraces] = useState<ReasoningBlock[]>([]);
  const [capabilities, setCapabilities] = useState<MerlinEvent[]>([]);
  const [isTauri, setIsTauri] = useState(false);
  const [sandboxLogs, setSandboxLogs] = useState<string[]>([
    "> MCP Registry initialized.",
    "> NATS Event Mesh listening...",
  ]);

  const [telemetry, setTelemetry] = useState<TelemetryState>({
    cpu: 0,
    ram: 0,
    vram: 7.4,
    vramTotal: 12,
    temperature: 68,
    tokensPerSec: 0,
    latencyMs: 0,
    history: Array.from({ length: 20 }, (_, i) => ({
      time: i,
      cpu: 0,
      ram: 0,
    })),
  });

  const chatScrollRef = useRef<HTMLDivElement>(null);
  // Track whether a request_human message was already shown for the current turn,
  // so we don't overwrite it with the generic "done" summary from task.completed.
  const shownRequestHumanRef = useRef(false);
  // Tracks whether a thinking session is currently active — used in the SSE closure
  // because `isThinking` state is stale in the [] useEffect closure.
  const isThinkingRef = useRef(false);

  useEffect(() => {
    if (window.__TAURI_INTERNALS__) {
      setIsTauri(true);
    }

    const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const eventSource = new EventSource(`${apiUrl}/api/stream`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as MerlinEvent;

        if (
          data.type === "agent.thinking" ||
          data.type === "agent.streaming"
        ) {
          setIsWaitingForHuman(false);
          if (!isThinkingRef.current) {
            isThinkingRef.current = true;
            setThinkingStart((prev) => prev !== null ? prev : Date.now());
            setLiveThinkingText("");
          }
          setIsThinking(true);

          // Accumulate live token stream for the thinking display
          if (data.type === "agent.thinking") {
            setLiveThinkingText((prev) => prev + (data.payload.text || ""));
          }

          // For streaming (narrative) events, inspect the text for tool-call JSON.
          // The LLM emits raw JSON like { "tool": "done", "args": {...} } as its
          // "response" before tool execution. This is internal scaffolding — suppress
          // it from the trace (the agent.tool_start ACTION block covers it visually).
          if (data.type === "agent.streaming") {
            const text = (data.payload.text || "").trim();
            try {
              const parsed = JSON.parse(text);
              if (parsed && typeof parsed.tool === "string") {
                // It's a tool-call JSON blob — handle request_human specially,
                // then bail out: don't add a narrative trace block.
                if (
                  parsed.tool === "request_human" &&
                  parsed.args?.question &&
                  !shownRequestHumanRef.current
                ) {
                  shownRequestHumanRef.current = true;
                  const msgId = data.id + "_rh";
                  setMessages((prev) => {
                    if (prev.some((m) => m.id === msgId)) return prev;
                    return [
                      ...prev,
                      {
                        id: msgId,
                        role: "assistant",
                        content: parsed.args.question,
                        timestamp: new Date(data.timestamp).toLocaleTimeString(
                          [],
                          { hour: "2-digit", minute: "2-digit" }
                        ),
                      },
                    ];
                  });
                }
                // Skip creating a narrative trace block for all tool-call JSON
                return;
              }
            } catch {
              // Not JSON — fall through to add as normal narrative trace
            }
          }

          setTraces((prev) => {
            const newTraces = [...prev];
            const blockType =
              data.type === "agent.thinking" ? "thinking" : "narrative";
            const lastTrace = newTraces[newTraces.length - 1];

            if (
              lastTrace &&
              lastTrace.type === blockType &&
              lastTrace.status === "pending"
            ) {
              newTraces[newTraces.length - 1] = {
                ...lastTrace,
                content: lastTrace.content + (data.payload.text || ""),
              };
              return newTraces;
            } else {
              if (lastTrace && lastTrace.status === "pending") {
                newTraces[newTraces.length - 1] = {
                  ...lastTrace,
                  status: "success",
                };
              }
              newTraces.push({
                id: data.id + Math.random().toString(),
                type: blockType,
                content: data.payload.text || "...",
                timestamp: new Date(data.timestamp).toLocaleTimeString([], {
                  hour12: false,
                }),
                status: "pending",
              });
              return newTraces.slice(-50);
            }
          });
        }

        if (data.type === "agent.tool_start") {
          // Each tool_start marks the end of a thinking phase — reset so the next
          // thinking phase gets a fresh start time and live text.
          isThinkingRef.current = false;
          setLiveThinkingText("");

          // If the agent is calling request_human, surface the question as a chat message
          if (data.payload.tool === "request_human") {
            const question =
              data.payload.args?.question ||
              data.payload.question ||
              data.payload.input;
            if (question) {
              shownRequestHumanRef.current = true;
              setMessages((prev) => {
                if (prev.some((m) => m.id === data.id)) return prev;
                return [
                  ...prev,
                  {
                    id: data.id,
                    role: "assistant",
                    content: typeof question === "string" ? question : JSON.stringify(question),
                    timestamp: new Date(data.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    }),
                  },
                ];
              });
            }
            // Agent is now suspended waiting for human input
            setIsWaitingForHuman(true);
            setIsThinking(false);
            setThinkingStart(null);
          }

          setTraces((prev) => {
            const newTraces = [...prev];
            const lastTrace = newTraces[newTraces.length - 1];
            if (lastTrace && lastTrace.status === "pending") {
              // If the last trace is a narrative that accumulated to valid tool-call JSON,
              // remove it retroactively — the ACTION block covers it visually.
              if (lastTrace.type === "narrative") {
                try {
                  const parsed = JSON.parse(lastTrace.content.trim());
                  if (parsed && typeof parsed.tool === "string") {
                    newTraces.pop();
                  } else {
                    newTraces[newTraces.length - 1] = { ...lastTrace, status: "success" };
                  }
                } catch {
                  newTraces[newTraces.length - 1] = { ...lastTrace, status: "success" };
                }
              } else {
                newTraces[newTraces.length - 1] = { ...lastTrace, status: "success" };
              }
            }
            // `done` is not a user-visible tool call — it's just the signal that the
            // task is finished. The result surfaces as a message via task.completed.
            if (data.payload.tool === "done") {
              return newTraces;
            }
            newTraces.push({
              id: data.id,
              type: "action",
              content: formatToolCall(data.payload.tool, data.payload.args),
              timestamp: new Date(data.timestamp).toLocaleTimeString([], {
                hour12: false,
              }),
              status: "pending",
            });
            return newTraces.slice(-50);
          });
          setSandboxLogs((prev) =>
            [...prev, `RUNNING: ${data.payload.tool}`].slice(-50)
          );
        }

        if (data.type === "agent.tool_end") {
          // `done` is not a real tool call — skip adding an observation for it.
          if (data.payload.tool === "done") {
            setSandboxLogs((prev) => [...prev, `DONE: done`].slice(-50));
          } else {
            setTraces((prev) => {
              const newTraces = [...prev];
              const lastTrace = newTraces[newTraces.length - 1];
              if (lastTrace && lastTrace.status === "pending") {
                newTraces[newTraces.length - 1] = {
                  ...lastTrace,
                  status: "success",
                };
              }
              newTraces.push({
                id: data.id,
                type: "observation",
                content: `Result: ${data.payload.result}`,
                timestamp: new Date(data.timestamp).toLocaleTimeString([], {
                  hour12: false,
                }),
                status: "success",
              });
              return newTraces.slice(-50);
            });
            setSandboxLogs((prev) =>
              [...prev, `DONE: ${data.payload.tool}`].slice(-50)
            );
          }
        }

        if (data.type === "task.completed") {
          isThinkingRef.current = false;
          setIsThinking(false);
          setThinkingStart(null);
          setLiveThinkingText("");
          setIsWaitingForHuman(false);
          // Only surface the done() summary if we didn't already show a request_human question
          if (!shownRequestHumanRef.current) {
            setMessages((prev) => {
              if (prev.some((m) => m.id === data.id)) return prev;
              return [
                ...prev,
                {
                  id: data.id,
                  role: "assistant",
                  content:
                    typeof data.payload.result === "string"
                      ? data.payload.result
                      : JSON.stringify(data.payload.result),
                  timestamp: new Date(data.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  }),
                },
              ];
            });
          }
          // Reset the flag for the next turn
          shownRequestHumanRef.current = false;
        }

        if (
          data.type === "action.failed" ||
          data.type === "task.failed"
        ) {
          setIsThinking(false);
          setThinkingStart(null);
          setIsWaitingForHuman(false);
          setMessages((prev) => {
            if (prev.some((m) => m.id === data.id)) return prev;
            return [
              ...prev,
              {
                id: data.id,
                role: "assistant",
                content: `Error: ${data.payload.reason}`,
                timestamp: new Date(data.timestamp).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                }),
              },
            ];
          });
        }

        if (
          data.type === "capability.gap" ||
          data.type === "improvement.queued" ||
          data.type === "improvement.deployed"
        ) {
          setCapabilities((prev) => [data, ...prev].slice(0, 20));
        }

        if (
          data.type === "system.heartbeat" &&
          !window.__TAURI_INTERNALS__
        ) {
          setTelemetry((prev) => {
            const ramPct = data.payload.ram_usage || 0;
            const cpuPct = data.payload.cpu_usage || 0;
            const newHistory = [
              ...prev.history.slice(1),
              {
                time: prev.history[prev.history.length - 1].time + 1,
                cpu: cpuPct,
                ram: ramPct,
              },
            ];
            return {
              ...prev,
              cpu: cpuPct,
              ram: ramPct,
              vram: data.payload.vram_used ?? prev.vram,
              vramTotal: data.payload.vram_total ?? prev.vramTotal,
              temperature: data.payload.temperature ?? prev.temperature,
              tokensPerSec: data.payload.tokens_per_sec ?? prev.tokensPerSec,
              latencyMs: data.payload.latency_ms ?? prev.latencyMs,
              history: newHistory,
            };
          });
        }
      } catch (err) {
        console.error("Failed to parse event", err);
      }
    };

    let interval: ReturnType<typeof setInterval> | undefined;
    if (window.__TAURI_INTERNALS__) {
      interval = setInterval(async () => {
        try {
          const info = await invoke<SystemInfo>("get_system_info");
          const ramPct = (info.used_memory / info.total_memory) * 100;
          setTelemetry((prev) => {
            const newHistory = [
              ...prev.history.slice(1),
              {
                time: prev.history[prev.history.length - 1].time + 1,
                cpu: info.cpu_usage,
                ram: ramPct,
              },
            ];
            return { ...prev, cpu: info.cpu_usage, ram: ramPct, history: newHistory };
          });
        } catch (err) {
          console.error("Failed to fetch system info", err);
        }
      }, 2000);
    }

    return () => {
      eventSource.close();
      if (interval) clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, traces, isThinking]);

  const handleSend = useCallback(async () => {
    if (!input.trim()) return;
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    setMessages((prev) => [...prev, userMsg]);
    if (!isWaitingForHuman) {
      setTraces([]);  // Only clear traces for new tasks, not human responses
    }
    setLiveThinkingText("");
    shownRequestHumanRef.current = false;
    setIsWaitingForHuman(false);
    setInput("");
    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      await fetch(
        `${apiUrl}/api/task?description=${encodeURIComponent(userMsg.content)}`,
        { method: "POST" }
      );
    } catch (err) {
      console.error("Failed to submit task", err);
    }
  }, [input, isWaitingForHuman]);

  return {
    messages,
    input,
    setInput,
    isThinking,
    thinkingStart,
    liveThinkingText,
    isWaitingForHuman,
    traces,
    capabilities,
    telemetry,
    isTauri,
    sandboxLogs,
    chatScrollRef,
    handleSend,
  };
}
