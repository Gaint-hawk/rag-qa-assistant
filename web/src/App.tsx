import { FormEvent, useEffect, useRef, useState } from "react";

type Citation = { source: string; cited_text: string };
type Source = { source: string; score: number };
type Doc = { id: string; title: string; chunks: number };
type Turn =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; citations: Citation[]; sources: Source[] }
  | { role: "error"; text: string };

const API = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  const loadDocs = () => request<Doc[]>("/documents").then(setDocs).catch(() => setDocs([]));
  useEffect(() => void loadDocs(), []);
  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [turns]);

  async function onUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        await request("/documents", { method: "POST", body: form });
      }
      await loadDocs();
    } catch (e) {
      setTurns((t) => [...t, { role: "error", text: (e as Error).message }]);
    } finally {
      setUploading(false);
    }
  }

  async function onRemove(id: string) {
    await request(`/documents/${id}`, { method: "DELETE" });
    await loadDocs();
  }

  async function onAsk(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (q.length < 3 || busy) return;
    setQuestion("");
    setTurns((t) => [...t, { role: "user", text: q }]);
    setBusy(true);
    try {
      const res = await request<{ answer: string; citations: Citation[]; sources: Source[] }>("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      setTurns((t) => [...t, { role: "assistant", text: res.answer, citations: res.citations, sources: res.sources }]);
    } catch (err) {
      setTurns((t) => [...t, { role: "error", text: (err as Error).message }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="layout">
      <aside>
        <h1>RAG Q&amp;A</h1>
        <label className="upload">
          {uploading ? "Uploading…" : "Upload .txt, .md or .pdf"}
          <input type="file" multiple accept=".txt,.md,.pdf" onChange={(e) => onUpload(e.target.files)} hidden />
        </label>
        <ul className="docs">
          {docs.length === 0 && <li className="muted">No documents yet</li>}
          {docs.map((d) => (
            <li key={d.id}>
              <span title={d.title}>{d.title}</span>
              <small>{d.chunks} chunks</small>
              <button aria-label={`Remove ${d.title}`} onClick={() => onRemove(d.id)}>
                ×
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main>
        <div className="chat">
          {turns.length === 0 && <p className="muted">Upload documents, then ask a question about them.</p>}
          {turns.map((t, i) => (
            <div key={i} className={`turn ${t.role}`}>
              <p>{t.text}</p>
              {t.role === "assistant" && t.citations.length > 0 && (
                <ol className="citations">
                  {t.citations.map((c, j) => (
                    <li key={j}>
                      <strong>{c.source}</strong> “{c.cited_text}”
                    </li>
                  ))}
                </ol>
              )}
            </div>
          ))}
          {busy && <div className="turn assistant muted">Thinking…</div>}
          <div ref={bottom} />
        </div>
        <form onSubmit={onAsk}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about your documents"
            aria-label="Question"
          />
          <button disabled={busy || question.trim().length < 3}>Ask</button>
        </form>
      </main>
    </div>
  );
}
