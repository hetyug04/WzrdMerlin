import { StatusBar } from "./components/StatusBar";
import { InferenceSidebar } from "./components/InferenceSidebar";
import { InteractionCanvas } from "./components/InteractionCanvas";
import { CapabilitySidebar } from "./components/CapabilitySidebar";
import { useMerlinSSE } from "./hooks/useMerlinSSE";

export default function App() {
  const {
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
  } = useMerlinSSE();

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-bg text-fg font-[var(--font-ui)] selection:bg-accent-blue/20">
      {/* Top: Status Bar */}
      <StatusBar telemetry={telemetry} isTauri={isTauri} />

      {/* Body: 3-column layout */}
      <div className="flex flex-1 min-h-0">
        {/* Left: Inference Sidebar */}
        <InferenceSidebar telemetry={telemetry} />

        {/* Center: Interaction Canvas */}
        <InteractionCanvas
          messages={messages}
          traces={traces}
          isThinking={isThinking}
          thinkingStart={thinkingStart}
          liveThinkingText={liveThinkingText}
          isWaitingForHuman={isWaitingForHuman}
          input={input}
          setInput={setInput}
          handleSend={handleSend}
          chatScrollRef={chatScrollRef}
        />

        {/* Right: Capability Sidebar */}
        <CapabilitySidebar
          capabilities={capabilities}
          sandboxLogs={sandboxLogs}
        />
      </div>
    </div>
  );
}
