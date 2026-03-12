import { useState, useCallback } from "react";
import {
  Folder,
  FolderOpen,
  FileText,
  ChevronRight,
  ChevronDown,
  X,
  RefreshCw,
} from "lucide-react";

const API_BASE = "http://localhost:8000";

interface FileEntry {
  name: string;
  type: "file" | "directory";
  size: number | null;
  modified: number;
}

interface TreeNode extends FileEntry {
  path: string;
  children?: TreeNode[];
  loaded?: boolean;
  expanded?: boolean;
}

function formatSize(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function joinPath(...parts: string[]): string {
  return parts.filter(Boolean).join("/");
}

export function FileExplorer() {
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [loading, setLoading] = useState(false);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [openFilePath, setOpenFilePath] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  /* ── Fetch directory listing ── */
  const fetchDir = useCallback(async (dirPath: string): Promise<TreeNode[]> => {
    const resp = await fetch(`${API_BASE}/api/files?path=${encodeURIComponent(dirPath)}`);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    return (data.entries as FileEntry[]).map((e) => ({
      ...e,
      path: joinPath(dirPath, e.name),
      children: e.type === "directory" ? [] : undefined,
      loaded: false,
      expanded: false,
    }));
  }, []);

  /* ── Load root on mount / refresh ── */
  const loadRoot = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const entries = await fetchDir("");
      setTree(entries);
    } catch (e: any) {
      setError(e.message ?? "Failed to connect");
    } finally {
      setLoading(false);
    }
  }, [fetchDir]);

  // Auto-load root on first render
  const [initialized, setInitialized] = useState(false);
  if (!initialized) {
    setInitialized(true);
    loadRoot();
  }

  /* ── Toggle directory ── */
  const toggleDir = useCallback(
    async (path: string) => {
      const toggle = async (nodes: TreeNode[]): Promise<TreeNode[]> => {
        const updated: TreeNode[] = [];
        for (const node of nodes) {
          if (node.path === path && node.type === "directory") {
            if (!node.loaded) {
              try {
                const children = await fetchDir(node.path);
                updated.push({ ...node, expanded: true, loaded: true, children });
              } catch {
                updated.push({ ...node, expanded: !node.expanded });
              }
            } else {
              updated.push({ ...node, expanded: !node.expanded });
            }
          } else if (node.children) {
            updated.push({ ...node, children: await toggle(node.children) });
          } else {
            updated.push(node);
          }
        }
        return updated;
      };
      setTree(await toggle(tree));
    },
    [tree, fetchDir]
  );

  /* ── Open file ── */
  const openFile = useCallback(async (path: string) => {
    setFileContent(null);
    setOpenFilePath(path);
    try {
      const resp = await fetch(
        `${API_BASE}/api/files/read?path=${encodeURIComponent(path)}`
      );
      if (!resp.ok) {
        setFileContent(`Error: ${await resp.text()}`);
        return;
      }
      setFileContent(await resp.text());
    } catch (e: any) {
      setFileContent(`Failed to load: ${e.message}`);
    }
  }, []);

  /* ── Recursive tree renderer ── */
  const renderNode = (node: TreeNode, depth: number = 0) => {
    const isDir = node.type === "directory";
    const isOpen = node.expanded;
    const indent = depth * 16;

    return (
      <div key={node.path}>
        <button
          onClick={() => (isDir ? toggleDir(node.path) : openFile(node.path))}
          className={`w-full flex items-center gap-1.5 px-2 py-[3px] text-left text-[12px] rounded-sm hover:bg-bg-tertiary transition-colors group ${
            openFilePath === node.path && !isDir
              ? "bg-accent-blue/10 text-accent-blue"
              : "text-fg-secondary"
          }`}
          style={{ paddingLeft: `${8 + indent}px` }}
        >
          {/* Chevron / spacer */}
          {isDir ? (
            isOpen ? (
              <ChevronDown className="w-3.5 h-3.5 shrink-0 text-fg-muted" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 shrink-0 text-fg-muted" />
            )
          ) : (
            <span className="w-3.5 shrink-0" />
          )}

          {/* Icon */}
          {isDir ? (
            isOpen ? (
              <FolderOpen className="w-3.5 h-3.5 shrink-0 text-accent-yellow" />
            ) : (
              <Folder className="w-3.5 h-3.5 shrink-0 text-accent-yellow" />
            )
          ) : (
            <FileText className="w-3.5 h-3.5 shrink-0 text-fg-muted" />
          )}

          {/* Name */}
          <span className="truncate flex-1 font-mono">{node.name}</span>

          {/* Size */}
          {!isDir && node.size !== null && (
            <span className="text-[10px] text-fg-muted font-mono shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
              {formatSize(node.size)}
            </span>
          )}
        </button>

        {/* Children */}
        {isDir && isOpen && node.children && (
          <div>{node.children.map((c) => renderNode(c, depth + 1))}</div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border">
        <span className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider">
          Workspace Files
        </span>
        <button
          onClick={loadRoot}
          className="p-1 rounded hover:bg-bg-tertiary text-fg-muted hover:text-fg-secondary transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-2 text-[11px] text-accent-red bg-accent-red/5 border-b border-border">
          {error}
        </div>
      )}

      {/* Tree */}
      <div className="flex-1 overflow-y-auto no-scrollbar py-1">
        {tree.length === 0 && !loading && !error && (
          <p className="text-[11px] text-fg-muted px-4 py-3">
            No files found in workspace.
          </p>
        )}
        {tree.map((n) => renderNode(n, 0))}
      </div>

      {/* File Viewer */}
      {fileContent !== null && (
        <div className="border-t border-border flex flex-col max-h-[50%] min-h-[140px]">
          <div className="flex items-center justify-between px-4 py-1.5 border-b border-border bg-bg-secondary">
            <span className="text-[10px] font-mono text-fg-muted truncate flex-1">
              {openFilePath}
            </span>
            <button
              onClick={() => {
                setFileContent(null);
                setOpenFilePath("");
              }}
              className="p-0.5 rounded hover:bg-bg-tertiary text-fg-muted ml-2"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          <pre className="flex-1 overflow-auto no-scrollbar px-4 py-2 text-[11px] font-mono text-fg-secondary leading-relaxed whitespace-pre-wrap break-all bg-bg">
            {fileContent}
          </pre>
        </div>
      )}
    </div>
  );
}
