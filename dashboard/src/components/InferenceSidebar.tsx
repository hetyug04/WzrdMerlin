import { LineChart, Line, ResponsiveContainer } from "recharts";
import type { TelemetryState, Actor } from "../types";

interface InferenceSidebarProps {
  telemetry: TelemetryState;
  actors?: Actor[];
}

const DEFAULT_ACTORS: Actor[] = [
  { id: "1", name: "merlin-brain", status: "ACTIVE", currentTask: "Awaiting Input", meshEvent: "NATS:msg_ack" },
  { id: "2", name: "forage-mcp", status: "IDLE", meshEvent: "NATS:ready" },
  { id: "3", name: "coder-agent", status: "SLEEP" },
];

export function InferenceSidebar({ telemetry, actors = DEFAULT_ACTORS }: InferenceSidebarProps) {
  const vramPct = telemetry.vramTotal > 0
    ? (telemetry.vram / telemetry.vramTotal) * 100
    : 0;
  const ramPct = telemetry.ram;
  const isGttSpill = ramPct > 85;

  return (
    <aside className="w-[260px] shrink-0 h-full border-r border-border bg-bg-secondary flex flex-col overflow-hidden select-none">
      {/* Section: Hardware */}
      <div className="p-4 border-b border-border">
        <h3 className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider mb-4">
          Hardware
        </h3>

        {/* VRAM bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-fg-secondary">VRAM</span>
            <span className="text-xs font-mono text-fg-muted">
              {telemetry.vram.toFixed(1)}/{telemetry.vramTotal}GB
            </span>
          </div>
          <div className="h-2.5 rounded bg-bg-tertiary overflow-hidden">
            <div
              className="h-full rounded bg-accent-yellow transition-all duration-500"
              style={{ width: `${Math.min(vramPct, 100)}%` }}
            />
          </div>
        </div>

        {/* RAM bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-fg-secondary">RAM</span>
            <span className="text-xs font-mono text-fg-muted">
              {ramPct.toFixed(0)}%
              {isGttSpill && (
                <span className="ml-1.5 text-accent-orange">GTT</span>
              )}
            </span>
          </div>
          <div className="h-2.5 rounded bg-bg-tertiary overflow-hidden">
            <div
              className={`h-full rounded transition-all duration-500 ${
                isGttSpill ? "bar-hatched" : "bg-accent-blue"
              }`}
              style={{ width: `${Math.min(ramPct, 100)}%` }}
            />
          </div>
        </div>

        {/* Telemetry ticker */}
        <p className="text-[11px] font-mono text-fg-muted mt-3 leading-relaxed">
          {telemetry.tokensPerSec.toFixed(1)} t/s
          <span className="mx-1.5 text-border-strong">&middot;</span>
          {telemetry.latencyMs.toFixed(0)}ms lat
          <span className="mx-1.5 text-border-strong">&middot;</span>
          {telemetry.temperature}°C
        </p>
      </div>

      {/* Section: Sparklines */}
      <div className="p-4 border-b border-border space-y-3">
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] text-fg-muted">CPU</span>
            <span className="text-[11px] font-mono text-fg-secondary">{telemetry.cpu.toFixed(1)}%</span>
          </div>
          <div className="h-10">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry.history}>
                <Line
                  type="monotone"
                  dataKey="cpu"
                  stroke="var(--accent-blue)"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] text-fg-muted">RAM</span>
            <span className="text-[11px] font-mono text-fg-secondary">{telemetry.ram.toFixed(1)}%</span>
          </div>
          <div className="h-10">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry.history}>
                <Line
                  type="monotone"
                  dataKey="ram"
                  stroke="var(--accent-yellow)"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Section: Actors (XState) */}
      <div className="flex-1 overflow-y-auto no-scrollbar p-4">
        <h3 className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider mb-3">
          Actors
        </h3>
        <ul className="space-y-1.5">
          {actors.map((actor) => (
            <li
              key={actor.id}
              className="flex items-center gap-2.5 px-2.5 py-2 rounded-md hover:bg-bg-tertiary transition-colors"
            >
              <StatusDot status={actor.status} />
              <span className="text-xs font-mono text-fg-secondary flex-1 truncate">
                {actor.name}
              </span>
              <span className="text-[10px] font-mono text-fg-muted uppercase">
                {actor.status}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* Footer */}
      <div className="px-4 py-2.5 border-t border-border text-[10px] font-mono text-fg-muted text-center">
        DisCo OS v2.0
      </div>
    </aside>
  );
}

function StatusDot({ status }: { status: Actor["status"] }) {
  if (status === "ACTIVE" || status === "BUSY") {
    return (
      <span
        className={`w-2 h-2 rounded-full ${
          status === "ACTIVE" ? "bg-accent-green" : "bg-accent-blue"
        }`}
      />
    );
  }
  // IDLE / SLEEP = hollow
  return (
    <span className="w-2 h-2 rounded-full border border-fg-muted" />
  );
}
