import { Sparkles } from "lucide-react";
import { useState } from "react";
import type { MerlinEvent, Tool } from "../types";
import { FileExplorer } from "./FileExplorer";

interface CapabilitySidebarProps {
  capabilities: MerlinEvent[];
  sandboxLogs: string[];
}

const DEFAULT_TOOLS: Tool[] = [
  { name: "search", type: "base" },
  { name: "python_repl", type: "base" },
  { name: "file_read", type: "base" },
  { name: "file_write", type: "base" },
  { name: "shell_exec", type: "base" },
];

export function CapabilitySidebar({
  capabilities,
  sandboxLogs,
}: CapabilitySidebarProps) {
  const [tab, setTab] = useState<"tools" | "files">("tools");

  // Derive synthesized tools from deployed capabilities
  const synthesizedTools: Tool[] = capabilities
    .filter((c) => c.type === "improvement.deployed")
    .map((c) => ({
      name: c.payload.tool_name || c.payload.gap_description || "new-tool",
      type: "synthesized" as const,
      version: c.payload.version || "v1.0.0",
      isNew: true,
    }));

  const allTools = [...DEFAULT_TOOLS, ...synthesizedTools];

  return (
    <aside className="w-[260px] shrink-0 h-full border-l border-border bg-bg-secondary flex flex-col overflow-hidden select-none">
      {/* Tab switcher */}
      <div className="flex border-b border-border">
        <button
          onClick={() => setTab("tools")}
          className={`flex-1 py-2 text-[11px] font-semibold uppercase tracking-wider transition-colors ${
            tab === "tools"
              ? "text-fg border-b-2 border-accent-blue"
              : "text-fg-muted hover:text-fg-secondary"
          }`}
        >
          Tools
        </button>
        <button
          onClick={() => setTab("files")}
          className={`flex-1 py-2 text-[11px] font-semibold uppercase tracking-wider transition-colors ${
            tab === "files"
              ? "text-fg border-b-2 border-accent-blue"
              : "text-fg-muted hover:text-fg-secondary"
          }`}
        >
          Files
        </button>
      </div>

      {/* Tab panels */}
      {tab === "files" ? (
        <FileExplorer />
      ) : (
        <>
          {/* Tool Registry */}
          <div className="flex-1 overflow-y-auto no-scrollbar p-4">
            <h3 className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider mb-3">
              Tool Registry
            </h3>

            <div className="flex flex-wrap gap-1.5">
              {allTools.map((tool) => (
                <ToolBadge key={tool.name} tool={tool} />
              ))}
            </div>

            {/* Capability Gaps / Events */}
            {capabilities.length > 0 && (
              <div className="mt-6">
                <h3 className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider mb-3">
                  Alita Events
                </h3>
                <ul className="space-y-2">
                  {capabilities.slice(0, 10).map((cap, i) => (
                    <CapabilityEvent key={i} event={cap} />
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Sandbox Feed */}
          <div className="h-44 border-t border-border bg-bg flex flex-col">
            <div className="flex items-center justify-between px-4 py-2 border-b border-border">
              <span className="text-[10px] font-semibold text-fg-muted uppercase tracking-wider">
                Sandbox
              </span>
              <span className="text-[10px] font-mono text-fg-muted">
                {sandboxLogs.length} lines
              </span>
            </div>
            <div className="flex-1 overflow-y-auto no-scrollbar px-4 py-2">
              {sandboxLogs.map((line, i) => (
                <p
                  key={i}
                  className="text-[11px] font-mono text-fg-muted leading-relaxed truncate"
                >
                  {line}
                </p>
              ))}
            </div>
          </div>
        </>
      )}
    </aside>
  );
}

/* ── Tool Badge ── */

function ToolBadge({ tool }: { tool: Tool }) {
  const isBase = tool.type === "base";

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-mono border ${
        isBase
          ? "bg-bg-tertiary border-border text-fg-secondary"
          : "bg-accent-yellow/10 border-accent-yellow/30 text-accent-yellow"
      }`}
    >
      {tool.name}
      {tool.version && (
        <span className="text-[9px] text-fg-muted">{tool.version}</span>
      )}
      {tool.isNew && (
        <Sparkles className="w-2.5 h-2.5 text-accent-yellow" />
      )}
    </span>
  );
}

/* ── Capability Event ── */

function CapabilityEvent({ event }: { event: MerlinEvent }) {
  const typeLabel = event.type.split(".")[1] || event.type;

  const colorMap: Record<string, string> = {
    gap: "text-accent-orange border-accent-orange/20 bg-accent-orange/5",
    queued: "text-accent-blue border-accent-blue/20 bg-accent-blue/5",
    deployed: "text-accent-green border-accent-green/20 bg-accent-green/5",
  };

  const cls = colorMap[typeLabel] || "text-fg-muted border-border bg-bg-tertiary";

  return (
    <div
      className={`px-3 py-2 rounded-md border text-xs font-mono ${cls}`}
    >
      <span className="uppercase tracking-wider text-[10px] font-semibold block mb-0.5">
        {typeLabel}
      </span>
      <span className="text-fg-secondary text-[11px]">
        {event.payload.gap_description ||
          JSON.stringify(event.payload).substring(0, 60)}
      </span>
    </div>
  );
}
