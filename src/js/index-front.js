const frontEndElem = {
    burgerCloseBtn : document.querySelector('.burger-close-btn'),
    burgerSvg : document.querySelector('.burger-close-btn svg:nth-child(1)'),
    closeSvg : document.querySelector('.burger-close-btn svg:nth-child(2)'),
    chMenuModal : document.getElementById('ch-menu-modal'),
    sendQuestionBtn : document.querySelector('.send-question-btn'),
    questionTextarea : document.querySelector('.question-textarea'),
    tagsChoices : document.querySelectorAll('.tags-choices'),
    chatBoxMain : document.querySelector('.chat-box-main'),
    chatBubble : document.querySelectorAll('.chat'),
    chatLoadingAnimation : document.querySelector('.chat-loading-animation'),
    uploadKnowledgeBaseBtn : document.querySelector('#upload-knowledge-btn'),
    uploadKnowledgeBaseFileInput : document.getElementById('dropzone-file'),
    uploadKnowledgeBaseFileNameDisplay : document.getElementById('file-name'),
    viewDbRowsBtn: document.getElementById('view-db-rows-btn'),
    truncateDbBtn: document.getElementById('truncate-db-btn'),
};

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function renderDebugDetails(result) {
    const timeLogs = result?.timeLogs || {};
    const selectedChunks = Array.isArray(result?.selectedChunks) ? result.selectedChunks : [];

    const timeLogItems = Object.entries(timeLogs)
        .map(([key, value]) => `<li><strong>${escapeHtml(key)}</strong>: ${escapeHtml(value)} ms</li>`)
        .join('');

    const chunkItems = selectedChunks
        .map(chunk => {
            const previewText = (chunk.text || '').slice(0, 220);
            return `
                <li>
                    <strong>#${escapeHtml(chunk.rank)}</strong>
                    | label: <strong>${escapeHtml(chunk.label)}</strong>
                    | score: ${escapeHtml(chunk.hybridScore)}
                    | vec: ${escapeHtml(chunk.vectorScore)}
                    | lex: ${escapeHtml(chunk.lexicalScore)}
                    | kw: ${escapeHtml(chunk.keywordCoverageScore)}
                    | distance: ${escapeHtml(chunk.vectorDistance)}
                    <br />
                    <em>keywords:</em> ${escapeHtml(chunk.keywords)}
                    <br />
                    <em>text:</em> ${escapeHtml(previewText)}${(chunk.text || '').length > 220 ? '...' : ''}
                </li>
            `;
        })
        .join('');

    return `
        <details class="mt-4">
            <summary><strong>Debug: Time Logs & Selected Chunks</strong></summary>
            <div class="mt-2">
                <h4><strong>Time Logs</strong></h4>
                ${timeLogItems ? `<ul>${timeLogItems}</ul>` : '<p>No timing logs available.</p>'}
                <h4><strong>Selected Chunks (${selectedChunks.length})</strong></h4>
                ${chunkItems ? `<ol>${chunkItems}</ol>` : '<p>No selected chunks available.</p>'}
            </div>
        </details>
    `;
}



frontEndElem.burgerCloseBtn.addEventListener('click', (e) => {
    e.preventDefault();

    if(frontEndElem.burgerCloseBtn.classList.contains('ch-closed')) {
        frontEndElem.burgerCloseBtn.classList.replace('ch-closed', 'ch-opened');
        frontEndElem.burgerSvg.classList.replace('block', 'hidden');
        frontEndElem.closeSvg.classList.replace('hidden', 'block');
        frontEndElem.chMenuModal.showPopover();
        return;
    }
    frontEndElem.burgerCloseBtn.classList.replace('ch-opened', 'ch-closed');
    frontEndElem.burgerSvg.classList.replace('hidden', 'block');
    frontEndElem.closeSvg.classList.replace('block', 'hidden');
    frontEndElem.chMenuModal.hidePopover();


});


frontEndElem.questionTextarea.addEventListener('input', () => {
    const hasText = frontEndElem.questionTextarea.value.trim().length > 0;
    const hasChatStarted = !!frontEndElem.chatBubble; 
    if (hasText) {
        frontEndElem.sendQuestionBtn.classList.remove('cursor-not-allowed');
        frontEndElem.sendQuestionBtn.removeAttribute('disabled');
    } else {
        frontEndElem.sendQuestionBtn.classList.add('cursor-not-allowed');
        frontEndElem.sendQuestionBtn.setAttribute('disabled', 'true');
    }

    if (hasText || hasChatStarted) {
        frontEndElem.chatBoxMain.classList.add('grow');
    } else {
        frontEndElem.chatBoxMain.classList.remove('grow');
    }
});


frontEndElem.sendQuestionBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    frontEndElem.sendQuestionBtn.disabled = true;

    const message = frontEndElem.questionTextarea.value.trim();

    if(message.length > 0) {
        frontEndElem.chatLoadingAnimation.classList.replace('hidden', 'block')
        const divElem = document.createElement('div');
        divElem.classList.add('chat', 'chat-end');
        divElem.innerHTML = `
            <div class="chat-bubble chat-bubble-primary text-wrap h-auto break-all">
                ${message}
            </div>
        `;
        frontEndElem.chatBoxMain.insertBefore(divElem, frontEndElem.chatLoadingAnimation);
        frontEndElem.sendQuestionBtn.disabled = true;
        frontEndElem.questionTextarea.value ='';
        frontEndElem.chatBoxMain.scrollTop = frontEndElem.chatBoxMain.scrollHeight;

        try {
            const response = await fetch('http://localhost:8080/ask-ai-with-vector', {
                method : 'POST',
                headers : {
                    'Content-Type' : 'application/json',
                    'x-auth-key' : 'SDP-AI-SERVER'
                },
                body : JSON.stringify({ message } )
            });

            const contentType = response.headers.get('content-type') || '';
            const isJson = contentType.includes('application/json');
            const result = isJson ? await response.json() : { message: await response.text() };

            if (!response.ok) {
                throw new Error(result?.message || result?.error || 'Request failed');
            }

            const newAIReply = document.createElement('div');
            newAIReply.classList.add('w-full', 'h-auto', 'px-4', 'py-6', 'leading-10', 'text-wrap', 'break-all', 'ai-response');

            const convertedMarkDown = marked.parse(result.response)
            const debugDetails = renderDebugDetails(result);
            newAIReply.innerHTML = `${convertedMarkDown}${debugDetails}`;
            frontEndElem.chatBoxMain.insertBefore(newAIReply, frontEndElem.chatLoadingAnimation);
            frontEndElem.chatBoxMain.scrollTop = frontEndElem.chatBoxMain.scrollHeight;
        } catch (error) {
            const errorBubble = document.createElement('div');
            errorBubble.classList.add('chat', 'chat-start');
            errorBubble.innerHTML = `
                <div class="chat-bubble chat-bubble-error text-wrap h-auto break-all">
                    ${escapeHtml(error?.message || 'Something went wrong. Please try again.')}
                </div>
            `;
            frontEndElem.chatBoxMain.insertBefore(errorBubble, frontEndElem.chatLoadingAnimation);
            frontEndElem.chatBoxMain.scrollTop = frontEndElem.chatBoxMain.scrollHeight;
        } finally {
            frontEndElem.chatLoadingAnimation.classList.replace('block', 'hidden');
            frontEndElem.sendQuestionBtn.disabled = false;
        }
        return;
    }

});



frontEndElem.questionTextarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {    
        e.preventDefault();
        frontEndElem.sendQuestionBtn.click();
    }
});


frontEndElem.uploadKnowledgeBaseFileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});


function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        console.log('File uploaded:', file.name);
        frontEndElem.uploadKnowledgeBaseFileNameDisplay.innerText = file.name;
        frontEndElem.uploadKnowledgeBaseFileNameDisplay.classList.remove('hidden');
    }
}

function validateFile(file) {
    if (file && file.type === "application/pdf") {
        return true;
    } else {
        return false;
    }
}


frontEndElem.uploadKnowledgeBaseFileInput.addEventListener('change', (e) => {
    const isPDF = validateFile(e.target.files[0]);
    const isFileInputNotEmpty = e.target.files.length > 0;
    if(isFileInputNotEmpty && isPDF) {
        frontEndElem.uploadKnowledgeBaseBtn.disabled = false;
        return
    }
    else {
        frontEndElem.uploadKnowledgeBaseBtn.disabled = true;
    }
});



frontEndElem.uploadKnowledgeBaseBtn.addEventListener('click', async(e) => {
    e.preventDefault();

    const pdfFile = frontEndElem.uploadKnowledgeBaseFileInput;
    if (!pdfFile.files[0]) return alert("Select a PDF first");

    const file = pdfFile.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload-knowledge-base', {
            method : 'POST',
            body : formData
        });

        if(!response.ok) {
            let errorMessage = 'Server Error';
            try {
                const errorData = await response.json();
                errorMessage = errorData.details || errorData.error || errorMessage;
            } catch {
                const fallbackText = await response.text();
                if (fallbackText) errorMessage = fallbackText;
            }
            throw new Error(errorMessage);
        }

        const responseData = await response.json();
        alert(responseData.message)
    }
    catch(e) {
        alert(e.message);
    }

    
})


frontEndElem.viewDbRowsBtn.addEventListener('click', async (e) => {
    e.preventDefault();

    try {
        const response = await fetch('/admin/db-rows?limit=20', {
            method: 'GET',
            headers: {
                'x-auth-key': 'SDP-AI-SERVER'
            }
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || 'Failed to fetch rows');
        }

        const formattedRows = data.rows.map((row) => ({
            id: row.id,
            label: row.label,
            keywords: row.keywords,
            text: (row.text || '').slice(0, 240)
        }));

        const dbViewElem = document.createElement('div');
        dbViewElem.classList.add('w-full', 'h-auto', 'px-4', 'py-6', 'text-wrap', 'break-all', 'ai-response');
        dbViewElem.innerHTML = `
            <h3><strong>DB Rows Preview</strong></h3>
            <p>Total rows: <strong>${data.total}</strong></p>
            <pre>${JSON.stringify(formattedRows, null, 2)}</pre>
        `;

        frontEndElem.chatBoxMain.insertBefore(dbViewElem, frontEndElem.chatLoadingAnimation);
        frontEndElem.chatBoxMain.scrollTop = frontEndElem.chatBoxMain.scrollHeight;
    } catch (error) {
        alert(error.message || 'Failed to fetch database rows.');
    }
});


frontEndElem.truncateDbBtn.addEventListener('click', async (e) => {
    e.preventDefault();

    const proceed = window.confirm('This will delete all rows in knowledge_base. Continue?');
    if (!proceed) return;

    try {
        const response = await fetch('/admin/db-truncate', {
            method: 'POST',
            headers: {
                'x-auth-key': 'SDP-AI-SERVER'
            }
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || 'Failed to truncate table');
        }

        alert(data.message || 'Database truncated.');
    } catch (error) {
        alert(error.message || 'Failed to truncate database.');
    }
});


// frontEndElem.tagsChoices.forEach( (tag) => {
//     tag.addEventListener('click', (e) => {
//         e.preventDefault();
//         const tagText = tag.textContent.trim();
//         const existingTags = frontEndElem.questionTextarea.value.split(',').map(t => t.trim()).filter(t => t.length > 0);