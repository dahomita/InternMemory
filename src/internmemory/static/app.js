(() => {
  const thread = document.getElementById("thread");
  const form = document.getElementById("ask-form");
  const messageInput = document.getElementById("message");
  const askButton = document.getElementById("ask-button");
  const includeTrace = document.getElementById("include-trace");
  const providerStatus = document.getElementById("provider-status");
  const confirmPanel = document.getElementById("confirm-panel");
  const confirmTitle = document.getElementById("confirm-title");
  const confirmBody = document.getElementById("confirm-body");
  const confirmMeta = document.getElementById("confirm-meta");
  const confirmSave = document.getElementById("confirm-save");
  const confirmDismiss = document.getElementById("confirm-dismiss");
  const notesList = document.getElementById("notes-list");
  const refreshNotes = document.getElementById("refresh-notes");

  let pendingConfirmation = null;

  function setBusy(busy) {
    askButton.disabled = busy;
    confirmSave.disabled = busy;
    messageInput.disabled = busy;
  }

  function clearThreadPlaceholder() {
    const empty = thread.querySelector(".empty-hint");
    if (empty) {
      empty.remove();
    }
  }

  function appendMessage(role, body, meta = "", extra = null) {
    clearThreadPlaceholder();
    const article = document.createElement("article");
    article.className = `message ${role}`;

    const roleEl = document.createElement("p");
    roleEl.className = "message-role";
    roleEl.textContent = role === "user" ? "You" : role === "error" ? "Error" : "InternMemory";
    article.appendChild(roleEl);

    const bodyEl = document.createElement("p");
    bodyEl.className = "message-body";
    bodyEl.textContent = body;
    article.appendChild(bodyEl);

    if (meta) {
      const metaEl = document.createElement("p");
      metaEl.className = "message-meta";
      metaEl.textContent = meta;
      article.appendChild(metaEl);
    }

    if (extra) {
      article.appendChild(extra);
    }

    thread.appendChild(article);
    thread.scrollTop = thread.scrollHeight;
    return article;
  }

  function showEmptyHint() {
    if (thread.children.length > 0) {
      return;
    }
    const hint = document.createElement("p");
    hint.className = "empty-hint";
    hint.textContent =
      "Ask about a note, a document, or a daily recap. If the agent drafts a note, you confirm it here before it is saved.";
    thread.appendChild(hint);
  }

  function hideConfirmation() {
    pendingConfirmation = null;
    confirmPanel.hidden = true;
    confirmTitle.textContent = "";
    confirmBody.textContent = "";
    confirmMeta.textContent = "";
  }

  function showConfirmation(pending) {
    pendingConfirmation = pending;
    confirmTitle.textContent = pending.title || "Untitled note";
    confirmBody.textContent = pending.content || "";
    const project =
      pending.project_id == null ? "no project" : `project ${pending.project_id}`;
    confirmMeta.textContent = `${pending.work_date} · ${project}`;
    confirmPanel.hidden = false;
  }

  function formatTrace(trace) {
    if (!Array.isArray(trace) || trace.length === 0) {
      return null;
    }
    const lines = trace.map((step) => {
      if (step.kind === "tool_call" && step.tool_call) {
        return `step ${step.step_number}: ${step.tool_call.tool_name}`;
      }
      if (step.kind === "final_answer") {
        return `step ${step.step_number}: final answer`;
      }
      return `step ${step.step_number}: ${step.kind}`;
    });
    const pre = document.createElement("pre");
    pre.className = "trace";
    pre.textContent = lines.join("\n");
    return pre;
  }

  async function readError(response) {
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") {
        return payload.detail;
      }
      return JSON.stringify(payload.detail || payload);
    } catch {
      return `Request failed (${response.status})`;
    }
  }

  async function probeAgent() {
    try {
      const response = await fetch("/agent/status");
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = await response.json();
      if (payload.available) {
        providerStatus.textContent = "Agent ready";
        providerStatus.className = "status ok";
        return;
      }
      providerStatus.textContent = "Agent offline — set GROQ_API_KEY";
      providerStatus.className = "status warn";
    } catch {
      providerStatus.textContent = "API unreachable";
      providerStatus.className = "status warn";
    }
  }

  async function loadNotes() {
    notesList.replaceChildren();
    try {
      const response = await fetch("/notes?limit=12");
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const notes = await response.json();
      if (!notes.length) {
        const empty = document.createElement("li");
        empty.className = "note-item";
        empty.textContent = "No notes yet.";
        notesList.appendChild(empty);
        return;
      }
      for (const note of notes) {
        const item = document.createElement("li");
        item.className = "note-item";

        const title = document.createElement("p");
        title.className = "note-title";
        title.textContent = note.title || "Untitled note";
        item.appendChild(title);

        const date = document.createElement("p");
        date.className = "note-date";
        date.textContent = note.work_date;
        item.appendChild(date);

        const snippet = document.createElement("p");
        snippet.className = "note-snippet";
        snippet.textContent =
          note.content.length > 140
            ? `${note.content.slice(0, 140)}…`
            : note.content;
        item.appendChild(snippet);
        notesList.appendChild(item);
      }
    } catch (error) {
      const item = document.createElement("li");
      item.className = "note-item";
      item.textContent = error.message || "Could not load notes.";
      notesList.appendChild(item);
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) {
      return;
    }

    appendMessage("user", message);
    messageInput.value = "";
    hideConfirmation();
    setBusy(true);

    try {
      const response = await fetch("/agent/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          include_trace: includeTrace.checked,
        }),
      });

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const payload = await response.json();
      const answer =
        payload.answer ||
        (payload.error ? `Run ended: ${payload.error}` : "No answer returned.");
      const metaParts = [
        `status ${payload.status}`,
        `${payload.step_count} steps`,
        `${payload.usage.input_tokens + payload.usage.output_tokens} tokens`,
      ];
      if (payload.abstained) {
        metaParts.push("abstained");
      }
      appendMessage(
        "assistant",
        answer,
        metaParts.join(" · "),
        includeTrace.checked ? formatTrace(payload.trace) : null,
      );

      if (payload.pending_confirmation) {
        showConfirmation(payload.pending_confirmation);
      }
      providerStatus.textContent = "Agent ready";
      providerStatus.className = "status ok";
    } catch (error) {
      appendMessage("error", error.message || "Request failed.");
      if (String(error.message || "").toLowerCase().includes("model provider")) {
        providerStatus.textContent = "Agent offline — set GROQ_API_KEY";
        providerStatus.className = "status warn";
      }
    } finally {
      setBusy(false);
      messageInput.focus();
    }
  });

  confirmSave.addEventListener("click", async () => {
    if (!pendingConfirmation) {
      return;
    }
    setBusy(true);
    try {
      const response = await fetch("/agent/confirm-note", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: pendingConfirmation.title || "",
          content: pendingConfirmation.content,
          work_date: pendingConfirmation.work_date,
          project_id: pendingConfirmation.project_id,
          confirmed: true,
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = await response.json();
      hideConfirmation();
      appendMessage(
        "assistant",
        `Saved note #${payload.note.id}: ${payload.note.title || "Untitled note"}`,
      );
      await loadNotes();
    } catch (error) {
      appendMessage("error", error.message || "Could not save note.");
    } finally {
      setBusy(false);
    }
  });

  confirmDismiss.addEventListener("click", () => {
    hideConfirmation();
    appendMessage("assistant", "Draft dismissed. Nothing was saved.");
  });

  refreshNotes.addEventListener("click", () => {
    loadNotes();
  });

  messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  showEmptyHint();
  probeAgent();
  loadNotes();
})();
