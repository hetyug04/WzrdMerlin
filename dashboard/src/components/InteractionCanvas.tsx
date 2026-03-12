import { useState, useEffect, useRef, type RefObject } from "react";
import { AnimatePresence, motion } from "motion/react";
import ReactMarkdown from "react-markdown";
import { Send, ChevronRight, Copy, Check, Zap } from "lucide-react";
import type { Message, ReasoningBlock } from "../types";

interface InteractionCanvasProps {
  messages: Message[];
  traces: ReasoningBlock[];
  isThinking: boolean;
  thinkingStart: number | null;
  liveThinkingText: string;
  isWaitingForHuman: boolean;
  input: string;
  setInput: (v: string) => void;
  handleSend: () => void;
  chatScrollRef: RefObject<HTMLDivElement | null>;
}

export function InteractionCanvas({
  messages,
  traces,
  isThinking,
  thinkingStart,
  liveThinkingText,
  isWaitingForHuman,
  input,
  setInput,
  handleSend,
  chatScrollRef,
}: InteractionCanvasProps) {
  return (
    <main className="flex-1 flex flex-col h-full min-w-0 bg-bg">
      <div
        ref={chatScrollRef}
        className="flex-1 overflow-y-auto no-scrollbar px-6 py-6 space-y-6"
      >
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}

        {/* Current turn: inline agent work stream */}
        <AnimatePresence>
          {(traces.length > 0 || isThinking) && (
            <AgentTurnBlock
              traces={traces}
              isThinking={isThinking}
              liveThinkingText={liveThinkingText}
              thinkingStart={thinkingStart}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Input bar */}
      <div className="border-t border-border px-6 py-4 bg-bg-secondary">
        <div className="flex items-center gap-2 border border-border rounded-lg bg-bg overflow-hidden focus-within:border-border-strong transition-colors">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={isWaitingForHuman ? "Type your response to Merlin..." : "Type a command or ask a question..."}
            className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-sm text-fg placeholder:text-fg-muted"
          />
          <button
            onClick={handleSend}
            className="btn-ghost border-none mr-1 hover:text-accent-blue"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </main>
  );
}

/* ── Message Bubble ── */

const PROSE_CLASSES = [
  "prose prose-sm max-w-none text-fg text-[13px] leading-relaxed",
  "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
  "[&_p]:leading-relaxed [&_p]:mb-3 [&_p:last-child]:mb-0",
  "[&_pre]:bg-bg-tertiary [&_pre]:rounded [&_pre]:px-3 [&_pre]:py-2 [&_pre]:overflow-x-auto [&_pre]:text-[12px]",
  "[&_code]:font-mono [&_code]:text-[12px] [&_code]:bg-bg-tertiary [&_code]:rounded [&_code]:px-1 [&_code]:py-0.5",
  "[&_a]:text-accent-blue [&_a]:underline [&_a]:underline-offset-2",
  "[&_strong]:font-semibold [&_ul]:pl-4 [&_ol]:pl-4 [&_li]:my-0.5",
  "[&_h1]:text-base [&_h1]:font-semibold [&_h1]:mt-4 [&_h1]:mb-2",
  "[&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:mb-1.5",
  "[&_h3]:text-sm [&_h3]:font-medium [&_h3]:mt-2 [&_h3]:mb-1",
  "[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-fg-muted [&_blockquote]:italic",
  "[&_hr]:border-border",
  "[&_table]:w-full [&_table]:text-sm [&_th]:text-left [&_th]:font-medium [&_th]:py-1.5 [&_th]:border-b [&_th]:border-border [&_td]:py-1.5 [&_td]:border-b [&_td]:border-border/50",
].join(" ");

function AgentAvatar() {
  return (
    <div className="w-5 h-5 rounded-full bg-accent-yellow/15 border border-accent-yellow/30 flex items-center justify-center shrink-0">
      <Zap className="w-2.5 h-2.5 text-accent-yellow" />
    </div>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%]">
          <div className="flex items-center justify-end gap-1.5 mb-1.5">
            <span className="text-[10px] font-mono text-fg-muted">{msg.timestamp}</span>
            <span className="text-[10px] font-medium text-fg-muted uppercase tracking-wider">You</span>
          </div>
          <div className="bg-bg-tertiary border border-border rounded-2xl rounded-tr-sm px-4 py-2.5 text-[13px] text-fg leading-relaxed">
            {msg.content}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col max-w-[85%]">
      <div className="flex items-center gap-2 mb-2">
        <AgentAvatar />
        <span className="text-[10px] font-medium text-fg-muted uppercase tracking-wider">Merlin</span>
        <span className="text-[10px] font-mono text-fg-muted">{msg.timestamp}</span>
      </div>
      <div className={`pl-7 ${PROSE_CLASSES}`}>
        <ReactMarkdown>{msg.content}</ReactMarkdown>
      </div>
    </div>
  );
}

/* ── Agent Turn Block – inline work stream ── */

function AgentTurnBlock({
  traces,
  isThinking,
  liveThinkingText,
  thinkingStart,
}: {
  traces: ReasoningBlock[];
  isThinking: boolean;
  liveThinkingText: string;
  thinkingStart: number | null;
}) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!thinkingStart) return;
    setElapsed(Math.floor((Date.now() - thinkingStart) / 1000));
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - thinkingStart) / 1000)),
      1000
    );
    return () => clearInterval(id);
  }, [thinkingStart]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.15 }}
      className="flex flex-col"
    >
      {/* Agent label row */}
      <div className="flex items-center gap-2 mb-3">
        <AgentAvatar />
        <span className="text-[10px] font-medium text-fg-muted uppercase tracking-wider">Merlin</span>
        {isThinking && (
          <span className="text-[10px] font-mono text-fg-muted tabular-nums">{elapsed}s</span>
        )}
      </div>

      {/* Inline trace stream */}
      <div className="pl-7 flex flex-col gap-1.5 border-l border-border ml-2.5">
        {traces.map((trace) => (
          <TraceItem key={trace.id} trace={trace} />
        ))}

        {/* Live token stream at the bottom */}
        {isThinking && <LiveThinkingLine text={liveThinkingText} />}

        {/* Bouncing dots fallback when thinking hasn't produced tokens yet */}
        {isThinking && !liveThinkingText && traces.length === 0 && (
          <div className="flex items-center gap-1 py-1 pl-3">
            {[0, 150, 300].map((delay) => (
              <span
                key={delay}
                className="w-1.5 h-1.5 rounded-full bg-fg-muted animate-bounce"
                style={{ animationDelay: `${delay}ms` }}
              />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

/* ── Trace Item ── */

function cleanThought(content: string): string {
  // Strip scaffolding lines like "[iteration 1] Deciding next action…"
  return content
    .split("\n")
    .filter((line) => !/^\[iteration\s+\d+\]/i.test(line.trim()))
    .join("\n")
    .trim();
}

function TraceItem({ trace }: { trace: ReasoningBlock }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (trace.type === "thinking") {
    // While a thinking block is still streaming (pending), LiveThinkingLine
    // is showing it live at the bottom. Don't duplicate it here.
    if (trace.status === "pending") return null;

    const cleaned = cleanThought(trace.content);
    if (!cleaned) return null;

    // Completed thinking block — collapsible, never truncated
    return (
      <div className="pl-3 py-0.5">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-[11px] text-fg-muted hover:text-fg-secondary transition-colors italic"
        >
          <ChevronRight
            className={`w-3 h-3 shrink-0 transition-transform duration-150 ${expanded ? "rotate-90" : ""}`}
          />
          <span>{expanded ? "hide thinking" : "thinking…"}</span>
        </button>
        {expanded && (
          <p className="mt-1.5 ml-4 text-[11px] text-fg-muted italic leading-relaxed">
            {cleaned}
          </p>
        )}
      </div>
    );
  }

  if (trace.type === "action") {
    return (
      <div className="group pl-1">
        <div className="flex items-start gap-1.5">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 shrink-0 mt-0.5 text-fg-muted hover:text-fg-secondary transition-colors"
            aria-label={expanded ? "Collapse" : "Expand"}
          >
            <ChevronRight
              className={`w-3 h-3 transition-transform duration-150 ${expanded ? "rotate-90" : ""}`}
            />
          </button>
          <div className="flex-1 min-w-0 flex items-baseline justify-between gap-3">
            <code className="text-[12px] font-mono text-fg-secondary break-all leading-relaxed">
              {trace.content}
            </code>
            <button
              onClick={() => handleCopy(trace.content)}
              className="opacity-0 group-hover:opacity-100 text-fg-muted hover:text-fg-secondary transition-all shrink-0"
              aria-label="Copy"
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            </button>
          </div>
        </div>
        {expanded && (
          <pre className="mt-1 ml-4 text-[11px] font-mono text-fg-muted bg-bg-tertiary rounded px-3 py-2 overflow-x-auto whitespace-pre-wrap border border-border">
            {trace.content}
          </pre>
        )}
      </div>
    );
  }

  if (trace.type === "observation") {
    const raw = trace.content.replace(/^Result:\s*/, "");
    const preview = raw.length > 100 ? raw.substring(0, 100) + "…" : raw;
    return (
      <div className="pl-5">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-[11px] text-fg-muted hover:text-fg-secondary transition-colors"
        >
          <ChevronRight
            className={`w-3 h-3 transition-transform duration-150 shrink-0 ${expanded ? "rotate-90" : ""}`}
          />
          <span className="font-mono">{expanded ? "hide result" : preview}</span>
        </button>
        {expanded && (
          <pre className="mt-1.5 ml-4 text-[11px] font-mono text-fg-secondary bg-bg-tertiary border border-border rounded px-3 py-2 whitespace-pre-wrap break-words max-h-64 overflow-y-auto no-scrollbar">
            {raw}
          </pre>
        )}
      </div>
    );
  }

  // Narrative — only render if there's actual non-JSON content.
  // Suppress anything that starts with { or [ since it's a streaming
  // tool-call JSON blob that will be removed retroactively by tool_start.
  if (trace.type === "narrative" && trace.content.trim()) {
    const trimmed = trace.content.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) return null;
    return (
      <p className="text-[11px] text-fg-muted leading-relaxed pl-3 py-0.5">
        {trimmed.length > 300 ? trimmed.substring(0, 300) + "…" : trimmed}
      </p>
    );
  }

  return null;
}

/* ── Live Thinking Line ── */

function LiveThinkingLine({ text }: { text: string }) {
  const ref = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [text]);

  if (!text) return null;

  return (
    <p
      ref={ref}
      className="text-[11px] text-fg-muted italic leading-relaxed pl-3 py-0.5 max-h-36 overflow-y-auto no-scrollbar"
    >
      {text}
      <span className="inline-block w-[5px] h-[10px] bg-fg-muted ml-0.5 align-middle animate-[blink_1s_step-end_infinite]" />
    </p>
  );
}
