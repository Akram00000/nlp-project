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

    // Process content for display
    let formattedContent = content;

    if (role === 'assistant') {
        // Convert markdown-style formatting
        formattedContent = renderMarkdown(content);

        // Style consensus labels
        formattedContent = styleConsensusLabels(formattedContent);
    } else {
        // For user messages, just escape HTML
        formattedContent = `<p>${escapeHtml(content)}</p>`;
    }

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

    // Initialize any details elements for proper toggling
    messageDiv.querySelectorAll('details').forEach(details => {
        details.addEventListener('toggle', () => {
            scrollToBottom();
        });
    });

    scrollToBottom();
}

/**
 * Simple markdown renderer for structured answers
 */
function renderMarkdown(text) {
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
    // Add warning icons to minority opinions or other markers if needed
    let styledHtml = html;
    styledHtml = styledHtml.replace(/⚠️/g, '<span class="warning-icon">⚠️</span>');
    return styledHtml;
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
// Source Modal Functions
// ============================================================

/**
 * Show source modal with full content
 * @param {string} sourceId - The chunk_id of the source to display
 */
function showSourceModal(sourceId) {
    // Find source in currentSources by chunk_id
    const source = currentSources.find(s => s.chunk_id === sourceId);

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
    }
});

// ============================================================
// Initialization
// ============================================================

// Focus input on load
queryInput.focus();
