/**
 * Islamic RAG Interface - JavaScript
 * Handles API communication and UI interactions
 */

// ============================================================
// State
// ============================================================

let isLoading = false;
let currentSources = [];

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
// Event Listeners
// ============================================================

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

// Auto-resize textarea
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
});

// ============================================================
// API Functions
// ============================================================

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    // Add user message
    addMessage(query, 'user');
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Show loading
    setLoading(true);

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Store sources for this response
        currentSources = data.sources || [];
        
        // Add assistant response
        addMessage(data.answer, 'assistant', currentSources.length > 0);

    } catch (error) {
        console.error('Error:', error);
        addMessage(
            'عذراً، حدث خطأ في الاتصال. يرجى المحاولة مرة أخرى.\n\nSorry, a connection error occurred. Please try again.',
            'assistant'
        );
    } finally {
        setLoading(false);
    }
}

// ============================================================
// UI Functions
// ============================================================

function addMessage(content, role, hasSources = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    // Format content with line breaks
    const formattedContent = content
        .split('\n')
        .map(line => `<p>${escapeHtml(line)}</p>`)
        .join('');

    let sourcesButton = '';
    if (hasSources && role === 'assistant') {
        sourcesButton = `
            <button class="sources-btn" onclick="showSources()">
                📖 عرض المصادر (${currentSources.length})
            </button>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-content">
            ${formattedContent}
            ${sourcesButton}
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showSources() {
    if (currentSources.length === 0) return;

    sourcesList.innerHTML = currentSources.map((source, index) => `
        <div class="source-card">
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

function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    loadingOverlay.classList.toggle('active', loading);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
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
// Initialization
// ============================================================

// Focus input on load
queryInput.focus();
