/**
 * Islamic RAG Interface - JavaScript
 * Handles streaming API communication, chat memory, and UI interactions
 */

// ============================================================
// State Management
// ============================================================

const state = {
    isLoading: false,
    isStreaming: false,
    abortController: null,
    currentSources: [],
    chatHistory: [],       // Full chat history for context
    messages: [],          // All messages with metadata
    editingMessageId: null,
    maxHistoryMessages: 10, // Token limit management - keep last N for API
};

// ============================================================
// DOM Elements
// ============================================================

const messagesContainer = document.getElementById('messages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const sourcesPanel = document.getElementById('sourcesPanel');
const sourcesList = document.getElementById('sourcesList');

// ============================================================
// Local Storage - Chat Memory Persistence
// ============================================================

const STORAGE_KEY = 'islamic_rag_chat_history';

function loadChatHistory() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            const data = JSON.parse(saved);
            state.messages = data.messages || [];
            state.chatHistory = data.chatHistory || [];
            
            // Restore messages to DOM
            restoreMessagesFromHistory();
        }
    } catch (e) {
        console.error('Failed to load chat history:', e);
    }
}

function saveChatHistory() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            messages: state.messages,
            chatHistory: state.chatHistory,
            savedAt: new Date().toISOString(),
        }));
    } catch (e) {
        console.error('Failed to save chat history:', e);
    }
}

function clearChatHistory() {
    state.messages = [];
    state.chatHistory = [];
    state.currentSources = [];
    localStorage.removeItem(STORAGE_KEY);
    
    // Reset DOM to welcome message only
    messagesContainer.innerHTML = `
        <div class="message assistant">
            <div class="message-content">
                <p>السلام عليكم ورحمة الله وبركاته 👋</p>
                <p>أنا مساعدك للبحث في الفقه الإسلامي. اسألني عن أي حكم شرعي وسأبحث لك في كتب الفقه والفتاوى والأحاديث.</p>
                <p class="hint">مثال: ما حكم صلاة الجماعة؟</p>
            </div>
        </div>
    `;
}

function restoreMessagesFromHistory() {
    // Clear existing messages except welcome
    const welcomeMsg = messagesContainer.querySelector('.message.assistant');
    messagesContainer.innerHTML = '';
    if (welcomeMsg && state.messages.length === 0) {
        messagesContainer.appendChild(welcomeMsg);
    }
    
    // Restore all messages
    state.messages.forEach(msg => {
        addMessageToDOM(msg.content, msg.role, msg.id, msg.sources, false);
    });
}

// ============================================================
// Event Listeners
// ============================================================

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (state.editingMessageId) {
            submitEdit();
        } else {
            sendQuery();
        }
    }
});

// Auto-resize textarea
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
});

// ============================================================
// Message ID Generation
// ============================================================

function generateMessageId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// ============================================================
// Streaming API Functions
// ============================================================

async function sendQuery(editedContent = null) {
    const query = editedContent || queryInput.value.trim();
    if (!query || state.isLoading) return;

    // Cancel any ongoing editing
    cancelEdit();

    // Create unique message ID
    const userMsgId = generateMessageId();
    const assistantMsgId = generateMessageId();

    // Add user message
    const userMessage = {
        id: userMsgId,
        role: 'user',
        content: query,
        timestamp: new Date().toISOString(),
    };
    state.messages.push(userMessage);
    state.chatHistory.push({ role: 'user', content: query });
    
    addMessageToDOM(query, 'user', userMsgId, null, true);
    
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Show loading state
    setLoading(true);
    state.isStreaming = true;
    
    // Create abort controller for stopping
    state.abortController = new AbortController();

    // Create assistant message placeholder
    const assistantMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        sources: [],
    };
    state.messages.push(assistantMessage);
    
    const assistantDiv = addMessageToDOM('', 'assistant', assistantMsgId, null, true, true);
    const contentDiv = assistantDiv.querySelector('.message-text');

    try {
        // Prepare history for API (limited to maxHistoryMessages)
        const historyForAPI = state.chatHistory.slice(-state.maxHistoryMessages);
        
        const response = await fetch('/api/query/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query,
                history: historyForAPI.slice(0, -1), // Exclude current query
            }),
            signal: state.abortController.signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let streamedContent = '';
        let statusDiv = null;

        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        switch (data.type) {
                            case 'status':
                                // Update status indicator
                                if (!statusDiv) {
                                    statusDiv = document.createElement('div');
                                    statusDiv.className = 'status-indicator';
                                    contentDiv.appendChild(statusDiv);
                                }
                                statusDiv.innerHTML = `
                                    <span class="status-icon">${getStatusIcon(data.status)}</span>
                                    <span class="status-text">${data.message}</span>
                                `;
                                scrollToBottom();
                                break;
                                
                            case 'meta':
                                // Language detection info
                                break;
                                
                            case 'sources':
                                // Sources received - store them
                                state.currentSources = data.sources || [];
                                assistantMessage.sources = state.currentSources;
                                // Update status to show sources found
                                if (statusDiv) {
                                    statusDiv.innerHTML = `
                                        <span class="status-icon">📚</span>
                                        <span class="status-text">تم العثور على ${data.total} مصدر</span>
                                    `;
                                }
                                break;
                                
                            case 'token':
                                // Remove status indicator when first token arrives
                                if (statusDiv) {
                                    statusDiv.remove();
                                    statusDiv = null;
                                }
                                // Append token to streaming content
                                streamedContent += data.content;
                                contentDiv.innerHTML = renderMarkdown(streamedContent);
                                scrollToBottom();
                                break;
                                
                            case 'done':
                                // Final processed answer
                                const finalAnswer = data.answer;
                                assistantMessage.content = finalAnswer;
                                contentDiv.innerHTML = renderMarkdown(styleConsensusLabels(finalAnswer));
                                
                                // Add sources button if we have sources
                                if (state.currentSources.length > 0) {
                                    const sourcesBtn = document.createElement('button');
                                    sourcesBtn.className = 'sources-btn';
                                    sourcesBtn.onclick = showSources;
                                    sourcesBtn.innerHTML = `📖 عرض المصادر (${state.currentSources.length})`;
                                    contentDiv.parentElement.appendChild(sourcesBtn);
                                }
                                
                                // Update chat history for context
                                state.chatHistory.push({ role: 'assistant', content: finalAnswer });
                                break;
                                
                            case 'error':
                                throw new Error(data.message);
                        }
                    } catch (parseError) {
                        console.error('Failed to parse SSE data:', parseError);
                    }
                }
            }
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Query aborted by user');
            contentDiv.innerHTML += '<p class="stopped-notice"><em>⏹️ تم إيقاف التوليد</em></p>';
            assistantMessage.content = contentDiv.textContent;
            state.chatHistory.push({ role: 'assistant', content: assistantMessage.content });
        } else {
            console.error('Error:', error);
            contentDiv.innerHTML = `
                <p>عذراً، حدث خطأ في الاتصال. يرجى المحاولة مرة أخرى.</p>
                <p>Sorry, a connection error occurred. Please try again.</p>
            `;
            assistantMessage.content = 'Error occurred';
        }
    } finally {
        setLoading(false);
        state.isStreaming = false;
        state.abortController = null;
        
        // Remove streaming indicator
        const streamingIndicator = assistantDiv.querySelector('.streaming-indicator');
        if (streamingIndicator) {
            streamingIndicator.remove();
        }
        
        // Save to localStorage
        saveChatHistory();
        scrollToBottom();
    }
}

function stopGeneration() {
    if (state.abortController) {
        state.abortController.abort();
    }
}

// ============================================================
// UI Functions
// ============================================================

function addMessageToDOM(content, role, messageId, sources = null, animate = true, isStreaming = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}${animate ? '' : ' no-animate'}`;
    messageDiv.setAttribute('data-message-id', messageId);

    let formattedContent = content;
    let actionsHtml = '';

    if (role === 'assistant') {
        formattedContent = content ? renderMarkdown(styleConsensusLabels(content)) : '';
        
        // Add streaming indicator if streaming
        const streamingIndicator = isStreaming ? '<span class="streaming-indicator">●</span>' : '';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${formattedContent}${streamingIndicator}</div>
            </div>
        `;
    } else {
        // User message with edit button
        formattedContent = `<p>${escapeHtml(content)}</p>`;
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${formattedContent}</div>
                <div class="message-actions">
                    <button class="edit-btn" onclick="startEdit('${messageId}')" title="تعديل">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;
    }

    messagesContainer.appendChild(messageDiv);
    
    if (animate) {
        scrollToBottom();
    }
    
    return messageDiv;
}

// ============================================================
// Edit Message Functions
// ============================================================

function startEdit(messageId) {
    // Find the message
    const msgIndex = state.messages.findIndex(m => m.id === messageId);
    if (msgIndex === -1) return;
    
    const message = state.messages[msgIndex];
    if (message.role !== 'user') return;
    
    state.editingMessageId = messageId;
    
    // Update input with message content
    queryInput.value = message.content;
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
    queryInput.focus();
    
    // Update send button to show edit mode
    sendBtn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"/>
        </svg>
    `;
    sendBtn.title = 'تأكيد التعديل';
    sendBtn.classList.add('edit-mode');
    
    // Highlight the message being edited
    const messageDiv = document.querySelector(`[data-message-id="${messageId}"]`);
    if (messageDiv) {
        messageDiv.classList.add('editing');
    }
    
    // Show cancel button
    showEditControls();
}

function cancelEdit() {
    if (!state.editingMessageId) return;
    
    const messageDiv = document.querySelector(`[data-message-id="${state.editingMessageId}"]`);
    if (messageDiv) {
        messageDiv.classList.remove('editing');
    }
    
    state.editingMessageId = null;
    queryInput.value = '';
    queryInput.style.height = 'auto';
    
    // Reset send button
    sendBtn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13"/>
        </svg>
    `;
    sendBtn.title = 'إرسال';
    sendBtn.classList.remove('edit-mode');
    
    hideEditControls();
}

function submitEdit() {
    if (!state.editingMessageId) return;
    
    const newContent = queryInput.value.trim();
    if (!newContent) return;
    
    const msgIndex = state.messages.findIndex(m => m.id === state.editingMessageId);
    if (msgIndex === -1) return;
    
    // Remove all messages from this point forward
    const removedMessages = state.messages.splice(msgIndex);
    
    // Also update chat history
    const historyIndex = Math.floor(msgIndex / 2) * 2; // Approximate mapping
    state.chatHistory = state.chatHistory.slice(0, historyIndex);
    
    // Remove from DOM
    removedMessages.forEach(msg => {
        const div = document.querySelector(`[data-message-id="${msg.id}"]`);
        if (div) div.remove();
    });
    
    // Reset edit state
    state.editingMessageId = null;
    sendBtn.classList.remove('edit-mode');
    hideEditControls();
    
    // Reset send button
    sendBtn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13"/>
        </svg>
    `;
    sendBtn.title = 'إرسال';
    
    // Send the edited query
    sendQuery(newContent);
}

function showEditControls() {
    let cancelBtn = document.getElementById('cancelEditBtn');
    if (!cancelBtn) {
        cancelBtn = document.createElement('button');
        cancelBtn.id = 'cancelEditBtn';
        cancelBtn.className = 'cancel-edit-btn';
        cancelBtn.onclick = cancelEdit;
        cancelBtn.innerHTML = '✕ إلغاء';
        
        const inputWrapper = document.querySelector('.input-wrapper');
        inputWrapper.parentElement.insertBefore(cancelBtn, inputWrapper.nextSibling);
    }
    cancelBtn.style.display = 'block';
}

function hideEditControls() {
    const cancelBtn = document.getElementById('cancelEditBtn');
    if (cancelBtn) {
        cancelBtn.style.display = 'none';
    }
}

/**
 * Simple markdown renderer for structured answers
 */
function renderMarkdown(text) {
    if (!text) return '';
    
    let html = text;

    // Escape HTML first (but preserve our special tags)
    html = html.replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Restore allowed HTML tags
    html = html.replace(/&lt;details&gt;/g, '<details>')
        .replace(/&lt;\/details&gt;/g, '</details>')
        .replace(/&lt;summary&gt;/g, '<summary>')
        .replace(/&lt;\/summary&gt;/g, '</summary>')
        .replace(/&lt;a href="#" class="source-link" data-source-id="([^"]+)" onclick="([^"]+)"&gt;/g, '<a href="#" class="source-link" data-source-id="$1" onclick="$2">')
        .replace(/&lt;a href="#source-([^"]+)" class="source-link"&gt;/g, '<a href="#source-$1" class="source-link">')
        .replace(/&lt;\/a&gt;/g, '</a>')
        .replace(/&lt;div class="minority-opinion-box"&gt;/g, '<div class="minority-opinion-box">')
        .replace(/&lt;\/div&gt;/g, '</div>')
        .replace(/&lt;span class="([^"]+)"&gt;/g, '<span class="$1">')
        .replace(/&lt;\/span&gt;/g, '</span>');

    // Headers (### Level 3, ## Level 2)
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

    // Bold (**text**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic (*text*)
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Blockquotes (> text)
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

    // Horizontal rules (---)
    html = html.replace(/^---$/gm, '<hr>');

    // Line breaks to paragraphs
    const paragraphs = html.split(/\n\n+/);
    html = paragraphs.map(p => {
        p = p.trim();
        if (p.startsWith('<h') || p.startsWith('<details') || p.startsWith('<div') || p.startsWith('<hr')) {
            return p;
        }
        if (p) {
            // Handle single line breaks within paragraphs
            p = p.replace(/\n/g, '<br>');
            return `<p>${p}</p>`;
        }
        return '';
    }).join('\n');

    return html;
}

/**
 * Apply styling to consensus labels
 */
function styleConsensusLabels(html) {
    if (!html) return '';
    let styledHtml = html;
    styledHtml = styledHtml.replace(/⚠️/g, '<span class="warning-icon">⚠️</span>');
    return styledHtml;
}

function showSources() {
    if (state.currentSources.length === 0) return;

    sourcesList.innerHTML = state.currentSources.map((source, index) => `
        <div class="source-card" onclick="showSourceModal('${source.chunk_id}')">
            <span class="source-type">${getSourceTypeLabel(source.type)}</span>
            <div class="source-meta">
                ${source.title ? `<strong>${escapeHtml(source.title)}</strong>` : ''}
                ${source.author ? ` - ${escapeHtml(source.author)}` : ''}
                ${source.scholar ? ` - ${escapeHtml(source.scholar)}` : ''}
            </div>
            <div class="source-content">${escapeHtml(source.content_preview)}</div>
        </div>
    `).join('');

    sourcesPanel.classList.add('active');
}

function toggleSources() {
    sourcesPanel.classList.toggle('active');
}

function getSourceTypeLabel(type) {
    const labels = {
        'fatwa': '📜 فتوى',
        'book': '📚 كتاب',
        'hadith': '📿 حديث',
    };
    return labels[type] || type;
}

/**
 * Get icon for status indicator
 */
function getStatusIcon(status) {
    const icons = {
        'analyzing': '🔍',
        'translating': '🌐',
        'retrieving': '📚',
        'generating': '✨',
    };
    return icons[status] || '⏳';
}

function setLoading(loading) {
    state.isLoading = loading;
    
    if (loading && state.isStreaming) {
        // Show stop button instead of send
        sendBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"/>
            </svg>
        `;
        sendBtn.title = 'إيقاف';
        sendBtn.onclick = stopGeneration;
        sendBtn.disabled = false;
        sendBtn.classList.add('stop-btn');
    } else {
        // Restore send button
        sendBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13"/>
            </svg>
        `;
        sendBtn.title = 'إرسال';
        sendBtn.onclick = sendQuery;
        sendBtn.disabled = loading;
        sendBtn.classList.remove('stop-btn');
    }
    
    // Don't show full overlay for streaming - it blocks the view
    // loadingOverlay.classList.toggle('active', loading);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close sources panel when clicking outside
sourcesPanel.addEventListener('click', (e) => {
    if (e.target === sourcesPanel) {
        toggleSources();
    }
});

// ============================================================
// Source Modal Functions
// ============================================================

/**
 * Show source modal with full content
 * @param {string} sourceId - The chunk_id of the source to display
 */
function showSourceModal(sourceId) {
    // Find source in currentSources by chunk_id
    const source = state.currentSources.find(s => s.chunk_id === sourceId);

    if (!source) {
        console.error('Source not found:', sourceId);
        return;
    }

    // Get modal elements
    const overlay = document.getElementById('sourceModalOverlay');
    const icon = document.getElementById('sourceIcon');
    const title = document.getElementById('sourceTitle');
    const badge = document.getElementById('sourceTypeBadge');
    const metadata = document.getElementById('sourceMetadata');
    const content = document.getElementById('sourceContent');

    // Set icon and badge based on source type
    const typeConfig = {
        'fatwa': { icon: '📜', label: 'فتوى', class: 'fatwa' },
        'hadith': { icon: '📿', label: 'حديث', class: 'hadith' },
        'book': { icon: '📚', label: 'كتاب', class: 'book' },
        'quran': { icon: '📖', label: 'قرآن', class: 'quran' },
    };

    const config = typeConfig[source.type] || { icon: '📄', label: source.type, class: '' };

    icon.textContent = config.icon;
    title.textContent = source.title || 'المصدر';
    badge.textContent = config.label;
    badge.className = `source-type-badge ${config.class}`;

    // Build metadata
    const metaItems = [];
    if (source.source_name) {
        metaItems.push(`<span class="meta-item">📌 ${escapeHtml(source.source_name)}</span>`);
    }
    if (source.scholar) {
        metaItems.push(`<span class="meta-item">👤 ${escapeHtml(source.scholar)}</span>`);
    }
    if (source.author) {
        metaItems.push(`<span class="meta-item">✍️ ${escapeHtml(source.author)}</span>`);
    }
    metaItems.push(`<span class="meta-item">📊 درجة الصلة: ${(source.score * 100).toFixed(0)}%</span>`);

    metadata.innerHTML = metaItems.join('');

    // Set content - use full_content if available
    const fullText = source.full_content || source.content_preview;
    content.innerHTML = formatSourceContent(fullText, source.type);

    // Show modal
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

/**
 * Close the source modal
 */
function closeSourceModal() {
    const overlay = document.getElementById('sourceModalOverlay');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
}

/**
 * Format source content based on type
 */
function formatSourceContent(text, type) {
    let html = escapeHtml(text);

    // Add line breaks
    html = html.replace(/\n/g, '<br>');

    // Style based on type
    if (type === 'hadith') {
        return `<div class="hadith-content">${html}</div>`;
    } else if (type === 'quran') {
        return `<div class="quran-content">${html}</div>`;
    }

    return `<div class="source-text">${html}</div>`;
}

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSourceModal();
        if (state.editingMessageId) {
            cancelEdit();
        }
    }
});

// ============================================================
// Header Actions
// ============================================================

function addHeaderActions() {
    const header = document.querySelector('.header');
    if (!header) return;
    
    // Check if actions already exist
    if (header.querySelector('.header-actions')) return;
    
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'header-actions';
    actionsDiv.innerHTML = `
        <button class="header-btn" onclick="clearChatHistory()" title="مسح المحادثة">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
        </button>
    `;
    header.appendChild(actionsDiv);
}

// ============================================================
// Initialization
// ============================================================

// Load chat history on page load
loadChatHistory();

// Add header actions
addHeaderActions();

// Focus input on load
queryInput.focus();
