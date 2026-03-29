import { useEffect, useRef, useState, type FormEvent } from "react";
import "./App.css";
import { departments } from "./data/departments";
import { chatWithCourseBot } from "./lib/api";
import type { ChatResponse, Provider, RecommendedCourse } from "./types/chat";

type UserMessage = {
  id: string;
  role: "user";
  content: string;
};

type AssistantMessage = {
  id: string;
  role: "assistant";
  data?: ChatResponse;
  errorText?: string;
};

type ChatMessage = UserMessage | AssistantMessage;

function App() {
  const [query, setQuery] = useState("");
  const [department, setDepartment] = useState("CS");
  const [provider, setProvider] = useState<Provider>("ollama");
  const [topK, setTopK] = useState(10);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setError("กรุณากรอกคำถามก่อน");
      return;
    }

    const userMessage: UserMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedQuery,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setError("");
    setQuery("");

    try {
      const data: ChatResponse = await chatWithCourseBot({
        query: trimmedQuery,
        department,
        top_k: topK,
        provider,
      });

      const assistantMessage: AssistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        data,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "เกิดข้อผิดพลาด";

      setError(errorMessage);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          errorText: `เกิดข้อผิดพลาด: ${errorMessage}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const currentDepartmentLabel =
    departments.find((item) => item.value === department)?.label || department;

  const hasConversation = messages.length > 0 || loading;

  const clearChat = () => {
    setMessages([]);
    setError("");
  };

  return (
    <div className="page-shell">
      <div className="page-bg" />
      <main className="container">
        <section className="grid-layout">
          <div className="panel">
            <div className="panel-header">
              <h2>ถามคำถาม</h2>
              <span>(แนะนำวิชาเรียนตามอาชีพ)</span>
            </div>

            <form className="form-stack" onSubmit={handleSubmit}>
              <label className="field">
                <span>สาขา</span>
                <select
                  value={department}
                  onChange={(e) => setDepartment(e.target.value)}
                >
                  {departments.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>ผู้ให้บริการโมเดล</span>
                <div className="toggle-row">
                  <button
                    type="button"
                    className={provider === "ollama" ? "chip active" : "chip"}
                    onClick={() => setProvider("ollama")}
                  >
                    Ollama
                  </button>
                  {/* <button
                    type="button"
                    className={provider === "hf" ? "chip active" : "chip"}
                    onClick={() => setProvider("hf")}
                  >
                    Hugging Face
                  </button> */}
                </div>
              </label>

              {/* <label className="field">
                <span>จำนวนวิชาที่ใช้ค้น (top_k)</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </label> */}

              <label className="field">
                <span>คำถาม</span>
                <textarea
                  rows={6}
                  placeholder="เช่น อยากเป็น Data Engineer ควรเรียนวิชาอะไร"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>

              <button className="submit-btn" type="submit" disabled={loading}>
                {loading ? (
                  <span className="submit-loading">
                    AI กำลังทำงาน
                    <span className="inline-dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                  </span>
                ) : (
                  "ถาม AI"
                )}
              </button>
            </form>

            {error && <div className="error-box">{error}</div>}
          </div>

          <div className="panel result-panel">
            <div className="panel-header">
              <h2>บทสนทนา</h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span>{currentDepartmentLabel}</span>
                <button
                  className="clear-chat-btn"
                  onClick={clearChat}
                  title="ล้างแชททั้งหมด"
                  disabled={loading}
                >
                  🗑️ ล้างแชท
                </button>
              </div>
            </div>

            <div className="result-panel-body">
              {messages.length === 0 && !loading ? (
                <div className="panel-empty-state">
                  <div>
                    <p>ยังไม่มีบทสนทนา</p>
                    <small>กรอกคำถาม แล้วกด “ถาม AI” เพื่อเริ่มสนทนา</small>
                  </div>
                </div>
              ) : (
                <div className="chat-thread">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`chat-row ${
                        message.role === "user" ? "user-row" : "assistant-row"
                      }`}
                    >
                      <div
                        className={`chat-bubble ${
                          message.role === "user" ? "user-bubble" : "assistant-bubble"
                        }`}
                      >
                        <div className="bubble-role">
                          {message.role === "user" ? "คุณ" : "AI"}
                        </div>

                        {message.role === "user" ? (
                          <p className="user-text">{message.content}</p>
                        ) : message.errorText ? (
                          <p className="user-text">{message.errorText}</p>
                        ) : message.data ? (
                          <AssistantResponse data={message.data} />
                        ) : null}
                      </div>
                    </div>
                  ))}

                  {loading && (
                    <div className="chat-row assistant-row">
                      <div className="chat-bubble assistant-bubble thinking-bubble">
                        <div className="bubble-role">AI</div>
                        <div className="thinking-line">
                          <span>AI กำลังทำงาน</span>
                          <div className="typing-dots" aria-label="AI กำลังทำงาน">
                            <span />
                            <span />
                            <span />
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={chatBottomRef} />
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function AssistantResponse({ data }: { data: ChatResponse }) {
  return (
    <div className="assistant-structured">
      <section className="answer-section">
        <div className="section-title">สรุปภาพรวม</div>
        <p className="assistant-paragraph">{data.summary}</p>
      </section>

      <section className="answer-section">
        <div className="section-title">เป้าหมายอาชีพ</div>
        <div className="tag-row">
          <span className="target-pill">{data.target_career || "ยังไม่ชัดเจน"}</span>
        </div>
      </section>

      {data.recommended_courses?.length > 0 && (
        <section className="answer-section">
          <div className="section-title">วิชาที่แนะนำ</div>
          <div className="recommend-list">
            {data.recommended_courses.map((course) => (
              <RecommendedCourseCard
                key={`${course.rank}-${course.course_code}-${course.course_name}`}
                course={course}
              />
            ))}
          </div>
        </section>
      )}

      {/* {data.learning_order?.length > 0 && (
        <section className="answer-section">
          <div className="section-title">ลำดับการเรียนที่แนะนำ</div>
          <ol className="order-list">
            {data.learning_order.map((item, index) => (
              <li key={`${item}-${index}`}>{item}</li>
            ))}
          </ol>
        </section>
      )} */}

      {data.career_paths?.length > 0 && (
        <section className="answer-section">
          <div className="section-title">สายอาชีพที่เกี่ยวข้อง</div>
          <div className="tag-row">
            {data.career_paths.map((item, index) => (
              <span key={`${item}-${index}`} className="tag-pill">
                {item}
              </span>
            ))}
          </div>
        </section>
      )}

      {data.note && (
        <section className="answer-section">
          <div className="section-title">หมายเหตุ</div>
          <p className="assistant-paragraph">{data.note}</p>
        </section>
      )}

      {/* {data.meta && (
        <div className="meta-row">
          {data.meta.provider && (
            <span className="meta-pill">Provider: {data.meta.provider}</span>
          )}
          {data.meta.model && (
            <span className="meta-pill">Model: {data.meta.model}</span>
          )}
          {data.meta.top_k && <span className="meta-pill">top_k: {data.meta.top_k}</span>}
        </div>
      )} */}

      {/* {!!data.answer_raw && (
        <details className="raw-answer-box">
          <summary>ดูคำตอบดิบจากโมเดล</summary>
          <pre>{data.answer_raw}</pre>
        </details>
      )} */}
    </div>
  );
}

function normalizeLevel(level?: string) {
  const value = (level || "กลาง").trim().toLowerCase();

  if (value.includes("สูง") || value.includes("high")) {
    return "high";
  }

  if (value.includes("ต่ำ") || value.includes("low")) {
    return "low";
  }

  return "medium";
}

function getLevelClass(level?: string) {
  const value = (level || "").toLowerCase().trim();

  if (value === "สูง" || value === "high") return "level-high";
  if (value === "ต่ำ" || value === "low") return "level-low";
  return "level-medium";
}

function RecommendedCourseCard({ course }: { course: RecommendedCourse }) {
  const [open, setOpen] = useState(false);
  const levelText = course.match_level || "กลาง";

  return (
    <div className={`course-card course-accordion ${open ? "open" : ""}`}>
      <div className="course-toggle-left">
        <div className="course-rank-badge">#{course.rank}</div>

        <div className="course-summary-main">
          <div className="course-title-row">
            <h3>{course.course_name || "ไม่ระบุชื่อวิชา"}</h3>
            <span className={`course-level-badge ${getLevelClass(levelText)}`}>
              {levelText}
            </span>
          </div>

          <div className="course-meta-row">
            <span className="course-code-badge">
              {course.course_code || "-"}
            </span>
          </div>
        </div>
      </div>

      <button
        type="button"
        className="course-toggle"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <span className="accordion-icon" aria-hidden="true">
          ▾
        </span>
      </button>

      {open && (
        <div className="course-details">
          {/* {!!course.reason && (
            <div className="course-detail-block">
              <div className="course-detail-label">เหตุผลที่แนะนำ</div>
              <p className="course-reason">{course.reason}</p>
            </div>
          )} */}

          {!!course.skills?.length && (
            <div className="course-detail-block">
              <div className="course-detail-label">ทักษะที่เกี่ยวข้อง</div>
              <div className="tag-row">
                {course.skills.map((skill, index) => (
                  <span key={`${skill}-${index}`} className="tag-pill">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {!!course.recommended_before_after && (
            <div className="course-detail-block">
              <div className="course-detail-label">คำแนะนำการเรียนต่อ</div>
              <div className="course-order-box">
                {course.recommended_before_after}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
