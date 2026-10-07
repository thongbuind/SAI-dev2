const API = "http://127.0.0.1:8000"

function getOrCreateUserId() {
  let uid = localStorage.getItem("sai_user_id")
  if (!uid) {
    uid = "u" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7)
    localStorage.setItem("sai_user_id", uid)
  }
  return uid
}
const USER_ID = getOrCreateUserId()

const chatEl         = document.getElementById("chat")
const welcomeOverlay = document.getElementById("welcome-overlay")
const inputWrapper   = document.getElementById("input-wrapper")

let isWelcome        = true
let currentSessionId = null
let currentMessages  = []
let currentShared    = false
let sidebarOpen      = false
let activeTab        = "mine"
let viewingShared    = false

let currentModel     = null

const sessionsMap    = {}

function newSessionId() {
  return "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

function animateChatInput(instant) {
  const sidebarW    = sidebarOpen ? 260 : 0
  const padding     = 24
  const maxW        = 808
  const vw          = window.innerWidth
  const totalW      = Math.min(maxW, vw - sidebarW - padding * 2)
  const targetLeft  = sidebarW + Math.max(padding, (vw - sidebarW - totalW) / 2)
  const targetRight = vw - targetLeft - totalW

  if (instant) {
    inputWrapper.style.transition = "none"
    inputWrapper.style.left  = targetLeft  + "px"
    inputWrapper.style.right = targetRight + "px"
    return
  }
  inputWrapper.style.transition = "left 0.42s cubic-bezier(0.34,1.56,0.64,1)"
  inputWrapper.style.left = targetLeft + "px"
  setTimeout(() => {
    inputWrapper.style.transition = "right 0.38s cubic-bezier(0.34,1.4,0.64,1)"
    inputWrapper.style.right = targetRight + "px"
  }, 80)
}

function toggleSidebar() {
  sidebarOpen = !sidebarOpen
  const sidebar = document.getElementById("sidebar")
  const btn = sidebarOpen
    ? document.getElementById("sidebar-open-btn")
    : document.getElementById("sidebar-toggle")
  if (btn) {
    const br = btn.getBoundingClientRect(), sr = sidebar.getBoundingClientRect()
    sidebar.style.transformOrigin =
      `${br.left + br.width / 2 - sr.left}px ${br.top + br.height / 2 - sr.top}px`
  }
  sidebar.classList.toggle("closed", !sidebarOpen)
  document.getElementById("main").classList.toggle("sidebar-open", sidebarOpen)
  document.getElementById("sidebar-open-btn")?.classList.toggle("hidden", sidebarOpen)
  isWelcome ? positionWelcomeInput(true) : animateChatInput(false)
}

function switchTab(tab) {
  activeTab = tab
  document.getElementById("tab-mine").classList.toggle("active", tab === "mine")
  document.getElementById("tab-community").classList.toggle("active", tab === "community")
  document.getElementById("history-list").innerHTML = ""
  tab === "mine" ? loadHistoryList() : loadCommunityList()
}

function renderHistoryList(sessions) {
  const list = document.getElementById("history-list")
  list.innerHTML = ""
  if (!sessions.length) {
    list.innerHTML = '<div class="history-empty">Chưa có lịch sử</div>'
    return
  }
  sessions.sort((a, b) => b.created_at - a.created_at)
  sessions.forEach(s => {
    const item = document.createElement("div")
    item.className  = "history-item" + (s.id === currentSessionId ? " active" : "")
    item.dataset.id = s.id

    const title = document.createElement("span")
    title.className   = "history-title"
    title.textContent = s.title || "Cuộc trò chuyện"
    title.onclick     = () => loadSession(s.id)

    const shareBtn = document.createElement("button")
    shareBtn.className = "history-action share-btn" + (s.shared ? " shared" : "")
    shareBtn.title     = s.shared ? "Đang chia sẻ — bấm để ẩn" : "Chia sẻ lên Community"
    shareBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="${s.shared ? "currentColor" : "none"}" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>`
    shareBtn.onclick   = async (e) => { e.stopPropagation(); await toggleShare(s.id) }

    const del = document.createElement("button")
    del.className = "history-action history-delete"
    del.title     = "Xoá"
    del.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>`
    del.onclick   = (e) => { e.stopPropagation(); deleteSession(s.id) }

    item.appendChild(title)
    item.appendChild(shareBtn)
    item.appendChild(del)
    list.appendChild(item)
  })
}

function renderCommunityList(sessions) {
  const list = document.getElementById("history-list")
  list.innerHTML = ""
  if (!sessions.length) {
    list.innerHTML = '<div class="history-empty">Chưa có cuộc trò chuyện nào được chia sẻ</div>'
    return
  }
  sessions.forEach(s => {
    const item = document.createElement("div")
    item.className  = "history-item community-item"
    item.dataset.id = s.id

    const meta = document.createElement("div")
    meta.className   = "community-meta"
    meta.textContent = "~" + s.user_id.slice(0, 8)

    const title = document.createElement("span")
    title.className   = "history-title"
    title.textContent = s.title || "Cuộc trò chuyện"

    item.appendChild(meta)
    item.appendChild(title)
    item.onclick = () => loadSharedSession(s.id, s.user_id)
    list.appendChild(item)
  })
}

async function loadHistoryList() {
  try {
    const res  = await fetch(`${API}/history?user_id=${USER_ID}`)
    const data = await res.json()
    renderHistoryList(data.sessions || [])
  } catch {
    document.getElementById("history-list").innerHTML =
      '<div class="history-empty">Không thể tải lịch sử</div>'
  }
}

async function loadCommunityList() {
  try {
    const res  = await fetch(`${API}/history/community`)
    const data = await res.json()
    renderCommunityList(data.sessions || [])
  } catch {
    document.getElementById("history-list").innerHTML =
      '<div class="history-empty">Không thể tải Community</div>'
  }
}

async function saveSession(snapId, snapMessages) {
  const sid  = snapId      ?? currentSessionId
  const msgs = snapMessages ?? currentMessages
  if (!sid || !msgs.length) return
  const title = msgs.find(m => m.role === "user")?.text?.slice(0, 40) || "Cuộc trò chuyện"
  try {
    await fetch(`${API}/history/save`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id:         sid,
        title,
        messages:   msgs,
        created_at: Date.now() / 1000,
        user_id:    USER_ID,
        shared:     currentShared,
      })
    })
    if (activeTab === "mine") loadHistoryList()
  } catch {}
}

async function toggleShare(sessionId) {
  try {
    const res  = await fetch(`${API}/history/${sessionId}/share?user_id=${USER_ID}`, { method: "PATCH" })
    const data = await res.json()
    if (sessionId === currentSessionId) {
      currentShared = data.shared
      updateShareIndicator()
    }
    loadHistoryList()
  } catch {}
}

async function loadSession(id) {
  try {
    const res  = await fetch(`${API}/history/${id}?user_id=${USER_ID}`)
    if (!res.ok) return
    const data = await res.json()
    if (!data.messages) return

    viewingShared    = false
    currentSessionId = id
    currentMessages  = data.messages
    currentShared    = data.shared ?? false
    sessionsMap[id]  = data.messages

    if (isWelcome) transitionToChat(false)
    chatEl.innerHTML = ""
    data.messages.forEach(m => renderMessage(m.role, m.text, m.model))
    updateShareIndicator()
    setReadOnly(false)
    document.querySelectorAll(".history-item").forEach(el =>
      el.classList.toggle("active", el.dataset.id === id))
  } catch {}
}

async function loadSharedSession(id, ownerUserId) {
  try {
    const res  = await fetch(`${API}/history/${id}?user_id=${USER_ID}`)
    if (!res.ok) return
    const data = await res.json()
    if (!data.messages) return

    viewingShared    = true
    currentSessionId = id
    currentMessages  = data.messages
    currentShared    = true

    if (isWelcome) transitionToChat(false)
    chatEl.innerHTML = ""
    data.messages.forEach(m => renderMessage(m.role, m.text, m.model))
    setReadOnly(true, ownerUserId)
    updateShareIndicator()
    document.querySelectorAll(".history-item").forEach(el =>
      el.classList.toggle("active", el.dataset.id === id))
  } catch {}
}

async function deleteSession(id) {
  try {
    await fetch(`${API}/history/${id}?user_id=${USER_ID}`, { method: "DELETE" })
    if (id === currentSessionId) newChat()
    loadHistoryList()
  } catch {}
}

function updateShareIndicator() {
  const btn = document.getElementById("shareToggleBtn")
  if (!btn) return
  if (viewingShared) { btn.style.display = "none"; return }
  btn.style.display = ""
  btn.title = currentShared ? "Đang chia sẻ — bấm để ẩn" : "Chia sẻ lên Community"
  btn.classList.toggle("shared", currentShared)
  btn.querySelector("svg").setAttribute("fill", currentShared ? "currentColor" : "none")
}

function setReadOnly(readonly, ownerUserId) {
  const textarea = document.getElementById("message")
  const sendBtn  = document.getElementById("sendBtn")
  const shareBtn = document.getElementById("shareToggleBtn")
  const banner   = document.getElementById("readonly-banner")

  textarea.disabled = readonly
  sendBtn.disabled  = readonly
  if (shareBtn) shareBtn.style.display = readonly ? "none" : ""

  if (readonly && banner) {
    banner.style.display = ""
    banner.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
      Đang xem chat của <b>~${(ownerUserId || "").slice(0, 8)}</b> — chỉ đọc
    `
  } else if (banner) {
    banner.style.display = "none"
  }
}

function newChat() {
  currentSessionId = newSessionId()
  currentMessages  = []
  currentShared    = false
  viewingShared    = false

  if (!isWelcome) {
    chatEl.innerHTML = ""
    chatEl.classList.remove("visible")
    inputWrapper.classList.remove("chat-pos")
    welcomeOverlay.classList.remove("hidden")
    const wc = document.getElementById("welcome-content")
    if (wc) { wc.style.display = ""; wc.style.visibility = "" }
    document.getElementById("brand").classList.remove("exit")
    document.getElementById("powered").classList.remove("exit")
    document.getElementById("message").placeholder = "Hỏi bất cứ điều gì..."
    isWelcome = true
    positionWelcomeInput(false)
    window.addEventListener("resize", onResizeWelcome)
  }
  setReadOnly(false)
  updateShareIndicator()
  document.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"))
}

function positionWelcomeInput(animate) {
  if (!isWelcome) return
  const sidebarW = sidebarOpen ? 260 : 0
  const vw       = window.innerWidth
  const centerX  = sidebarW + (vw - sidebarW) / 2
  inputWrapper.style.transition = animate
    ? "left 0.42s cubic-bezier(0.34,1.56,0.64,1)"
    : "none"
  if (animate) setTimeout(() => { inputWrapper.style.transition = "none" }, 500)
  inputWrapper.style.left      = centerX + "px"
  inputWrapper.style.transform = "translateX(-50%) translateY(-50%)"
}
function onResizeWelcome() { positionWelcomeInput(false) }
positionWelcomeInput(false)
window.addEventListener("resize", onResizeWelcome)

function transitionToChat(focusInput = true) {
  if (!isWelcome) return
  isWelcome = false
  document.getElementById("brand").classList.add("exit")
  document.getElementById("powered").classList.add("exit")
  const wc = document.getElementById("welcome-content")
  if (wc) wc.style.visibility = "hidden"
  requestAnimationFrame(() => {
    inputWrapper.style.width     = ""
    inputWrapper.style.left      = ""
    inputWrapper.style.transform = ""
    inputWrapper.classList.add("chat-pos")
    animateChatInput(true)
  })
  setTimeout(() => {
    welcomeOverlay.classList.add("hidden")
    if (wc) wc.style.display = "none"
    chatEl.classList.add("visible")
    document.getElementById("message").placeholder = "Nhập tin nhắn..."
    if (focusInput) document.getElementById("message").focus()
    window.removeEventListener("resize", onResizeWelcome)
  }, 520)
}

function makeFeedbackHTML() {
  return `
    <button class="feedback-btn" data-type="like" title="Hữu ích">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
    </button>
    <button class="feedback-btn" data-type="dislike" title="Không hữu ích">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>
    </button>
  `
}

function bindFeedbackBtns(container) {
  container.querySelectorAll(".feedback-btn").forEach(btn => {
    btn.addEventListener("click", function () {
      const already = this.classList.contains("active")
      const type    = this.dataset.type
      container.querySelectorAll(".feedback-btn")
        .forEach(b => b.classList.remove("active", "liked", "disliked"))
      if (!already) this.classList.add("active", type === "like" ? "liked" : "disliked")
    })
  })
}

function normalizeAIText(text) {
  const normalizedNewlines = String(text ?? "")
    .replace(/\r\n?/g, "\n")
    .replace(/\\r\\n|\\n|\\r/g, "\n")

  let result = ""
  let capitalizeNext = true

  for (let i = 0; i < normalizedNewlines.length; i++) {
    const char = normalizedNewlines[i]

    if (/\p{L}/u.test(char)) {
      result += capitalizeNext ? char.toLocaleUpperCase("vi-VN") : char
      capitalizeNext = false
      continue
    }

    result += char

    if (/\p{N}/u.test(char)) {
      capitalizeNext = false
    } else if (char === "\n") {
      capitalizeNext = true
    } else if (char === ".") {
      const previousChar = normalizedNewlines[i - 1] || ""
      const nextChar = normalizedNewlines[i + 1] || ""
      const isNumericSeparator = /\p{N}/u.test(previousChar) && /\p{N}/u.test(nextChar)
      if (!isNumericSeparator) capitalizeNext = true
    }
  }

  return result
}

function renderAIText(element, text) {
  element.replaceChildren()

  const boldPattern = /\*\*([\s\S]+?)\*\*/g
  let cursor = 0
  let match

  while ((match = boldPattern.exec(text)) !== null) {
    element.appendChild(document.createTextNode(text.slice(cursor, match.index)))

    const strong = document.createElement("strong")
    strong.textContent = match[1]
    element.appendChild(strong)

    cursor = match.index + match[0].length
  }

  element.appendChild(document.createTextNode(text.slice(cursor)))
}

function renderMessage(role, text, model) {
  if (role === "ai" && model) {
    const labelRow = document.createElement("div")
    labelRow.className   = "model-label-row"
    labelRow.textContent = model.replace(/\.pt$/i, "")
    chatEl.appendChild(labelRow)
  }

  const row = document.createElement("div")
  row.className = `message-row ${role}`

  if (role === "ai") {
    const avatar = document.createElement("div")
    avatar.className   = "ai-avatar"
    avatar.textContent = "SAI"
    row.appendChild(avatar)

    const msgBody = document.createElement("div")
    msgBody.className = "ai-msg-body"

    const bubble = document.createElement("div")
    bubble.className   = "bubble glass-bubble glass-content"
    renderAIText(bubble, normalizeAIText(text))
    msgBody.appendChild(bubble)

    const feedback = document.createElement("div")
    feedback.className = "feedback-row"
    feedback.innerHTML = makeFeedbackHTML()
    bindFeedbackBtns(feedback)
    msgBody.appendChild(feedback)

    row.appendChild(msgBody)
  } else {
    const bubble = document.createElement("div")
    bubble.className   = "bubble glass-bubble glass-content"
    bubble.textContent = text
    row.appendChild(bubble)
  }

  chatEl.appendChild(row)
  chatEl.scrollTop = chatEl.scrollHeight
}

function addMessage(role, text, model) {
  const normalizedText = role === "ai" ? normalizeAIText(text) : text
  currentMessages.push({ role, text: normalizedText, ...(model ? { model } : {}) })
  renderMessage(role, normalizedText, model)
}

function addAIMessageTypewriter(text, model) {
  const normalizedText = normalizeAIText(text)
  currentMessages.push({ role: "ai", text: normalizedText, ...(model ? { model } : {}) })

  if (model) {
    const labelRow = document.createElement("div")
    labelRow.className   = "model-label-row"
    labelRow.textContent = model.replace(/\.pt$/i, "")
    chatEl.appendChild(labelRow)
  }

  const row = document.createElement("div")
  row.className = "message-row ai"

  const avatar = document.createElement("div")
  avatar.className   = "ai-avatar"
  avatar.textContent = "SAI"
  row.appendChild(avatar)

  const msgBody = document.createElement("div")
  msgBody.className = "ai-msg-body"

  const bubble = document.createElement("div")
  bubble.className = "bubble glass-bubble glass-content"
  msgBody.appendChild(bubble)

  const feedback = document.createElement("div")
  feedback.className = "feedback-row"
  feedback.innerHTML = makeFeedbackHTML()
  bindFeedbackBtns(feedback)
  msgBody.appendChild(feedback)

  row.appendChild(msgBody)
  chatEl.appendChild(row)

  let i = 0
  ;(function type() {
    if (i < normalizedText.length) {
      i++
      renderAIText(bubble, normalizedText.slice(0, i))
      chatEl.scrollTop = chatEl.scrollHeight
      setTimeout(type, 18)
    }
  })()
}

function showTyping() {
  const row = document.createElement("div")
  row.className = "message-row ai"
  row.id        = "typing-row"
  const avatar = document.createElement("div")
  avatar.className   = "ai-avatar"
  avatar.textContent = "SAI"
  row.appendChild(avatar)
  const bubble = document.createElement("div")
  bubble.className = "bubble glass-bubble glass-content"
  bubble.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`
  row.appendChild(bubble)
  chatEl.appendChild(row)
  chatEl.scrollTop = chatEl.scrollHeight
}

function hideTyping() {
  document.getElementById("typing-row")?.remove()
}

async function send() {
  if (viewingShared) return
  const input = document.getElementById("message")
  const text  = input.value.trim()
  if (!text) return

  if (!currentSessionId) currentSessionId = newSessionId()
  if (isWelcome) transitionToChat()

  document.getElementById("sendBtn").disabled = true
  input.disabled = true

  addMessage("user", text)
  input.value        = ""
  input.style.height = "auto"
  showTyping()

  const sessionAtSend = currentSessionId

  try {
    const res = await fetch(`${API}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:    text,
        user_id:    USER_ID,
        model_name: currentModel,
      })
    })

    if (res.status === 429) throw new Error("rate_limit")

    const data = await res.json()
    const actualModel = data.model || currentModel

    if (currentSessionId === sessionAtSend) {
      hideTyping()
      addAIMessageTypewriter(data.reply, actualModel)
      await saveSession(sessionAtSend, [...currentMessages])
    }

  } catch (e) {
    if (currentSessionId === sessionAtSend) {
      hideTyping()
      addMessage("system",
        e.message === "rate_limit"
          ? "Gửi quá nhanh, vui lòng chờ một chút."
          : "Lỗi kết nối server. Vui lòng thử lại.")
    }
  }

  if (currentSessionId === sessionAtSend) {
    document.getElementById("sendBtn").disabled = false
    input.disabled = false
    input.focus()
  }
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() }
}

function autoResize(el) {
  el.style.height = "auto"
  el.style.height = Math.min(el.scrollHeight, 200) + "px"
}

let modelDropdownOpen = false

async function loadModelList() {
  const list = document.getElementById("model-list")
  if (!list) return
  try {
    const res  = await fetch(`${API}/models`)
    const data = await res.json()

    if (!currentModel) {
      currentModel = data.default || (data.models && data.models[0]) || null
    }

    renderModelList(data.models || [])

    if (currentModel) {
      document.getElementById("modelBtnLabel").textContent =
        currentModel.replace(/\.pt$/i, "")
    }
  } catch {
    list.innerHTML = '<div class="model-empty">Không thể tải danh sách</div>'
  }
}

function renderModelList(models) {
  const list = document.getElementById("model-list")
  list.innerHTML = ""
  if (!models.length) {
    list.innerHTML = '<div class="model-empty">Không có model nào</div>'
    return
  }
  models.forEach(name => {
    const item = document.createElement("button")
    item.className    = "model-item" + (name === currentModel ? " active" : "")
    item.dataset.name = name
    const label = name.replace(/\.pt$/i, "")
    item.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>
      <span>${label}</span>
      ${name === currentModel
        ? `<svg class="model-check" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"><path d="M5 12l5 5L19 7"/></svg>`
        : ""}
    `
    item.onclick = () => selectModel(name)
    list.appendChild(item)
  })
}

function selectModel(name) {
  currentModel = name
  document.getElementById("modelBtnLabel").textContent = name.replace(/\.pt$/i, "")
  renderModelList(
    Array.from(document.querySelectorAll(".model-item")).map(el => el.dataset.name)
  )
  closeModelDropdown()
}

function toggleModelDropdown(e) {
  e.stopPropagation()
  modelDropdownOpen = !modelDropdownOpen
  document.getElementById("model-dropdown").classList.toggle("hidden", !modelDropdownOpen)
  const chevron = document.getElementById("modelChevron")
  if (chevron) chevron.style.transform = modelDropdownOpen ? "rotate(180deg)" : ""
  if (modelDropdownOpen) loadModelList()
}

function closeModelDropdown() {
  modelDropdownOpen = false
  document.getElementById("model-dropdown")?.classList.add("hidden")
  const chevron = document.getElementById("modelChevron")
  if (chevron) chevron.style.transform = ""
}

document.addEventListener("click", e => {
  const wrap = document.getElementById("model-selector-wrap")
  if (modelDropdownOpen && wrap && !wrap.contains(e.target)) closeModelDropdown()
})

currentSessionId = newSessionId()
loadHistoryList()
loadModelList()
updateShareIndicator()
document.getElementById("sidebar").classList.add("closed")
