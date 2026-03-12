import type { TelemetryState } from "../types";

interface StatusBarProps {
  telemetry: TelemetryState;
  isTauri: boolean;
}

export function StatusBar({ telemetry, isTauri }: StatusBarProps) {
  return (
    <header className="h-10 flex items-center justify-between px-4 border-b border-border bg-bg-secondary shrink-0 select-none">
      {/* Left: identity */}
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-fg">merlin-v2.0</span>
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-accent-green">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full rounded-full bg-accent-green opacity-60 animate-ping" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-accent-green" />
          </span>
          Running
        </span>
        <span className="text-xs text-fg-muted font-mono">
          {telemetry.vram.toFixed(1)}/{telemetry.vramTotal}GB VRAM
        </span>
      </div>

      {/* Right: badges */}
      <div className="flex items-center gap-2">
        {/* VRAM badge */}
        <ResourceBadge
          label="VRAM"
          value={telemetry.vram}
          max={telemetry.vramTotal}
          unit="GB"
          warnAt={10}
          critAt={11.5}
        />
        {/* RAM badge */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-border text-xs">
          <span className="text-fg-muted">RAM</span>
          <span className="font-mono text-fg-secondary">{telemetry.ram.toFixed(0)}%</span>
          {telemetry.ram > 85 && (
            <span className="px-1 py-0.5 text-[10px] font-mono bg-accent-orange/15 text-accent-orange rounded">
              GTT
            </span>
          )}
        </div>
        {/* Inference badge */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-border text-xs">
          <span className="font-mono text-fg">{telemetry.tokensPerSec.toFixed(1)} t/s</span>
        </div>
        {/* Mode badge */}
        <div className="px-2.5 py-1 rounded-md border border-border text-[10px] font-mono text-fg-muted uppercase">
          {isTauri ? "native" : "web"}
        </div>
      </div>
    </header>
  );
}

function ResourceBadge({
  label,
  value,
  max,
  unit,
  warnAt,
  critAt,
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
  warnAt: number;
  critAt: number;
}) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  const barColor =
    value >= critAt
      ? "bg-accent-red"
      : value >= warnAt
        ? "bg-accent-yellow"
        : "bg-accent-blue";

  return (
    <div className="flex items-center gap-2 px-2.5 py-1 rounded-md border border-border text-xs min-w-[120px]">
      <span className="text-fg-muted">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-bg-tertiary overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className="font-mono text-fg-secondary whitespace-nowrap">
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
}
