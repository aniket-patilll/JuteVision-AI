const uploadBtn = document.getElementById('upload-btn');
const modal = document.getElementById('upload-modal');
const closeModal = document.getElementById('close-modal');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadList = document.getElementById('upload-list');
const currentCount = document.getElementById('current-count');

// Modal Logic
uploadBtn.addEventListener('click', () => {
    modal.classList.add('active');
    modal.style.pointerEvents = 'auto';
    modal.style.opacity = '1';
});

closeModal.addEventListener('click', () => {
    modal.classList.remove('active');
    modal.style.pointerEvents = 'none';
    modal.style.opacity = '0';
});

window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.classList.remove('active');
        modal.style.pointerEvents = 'none';
        modal.style.opacity = '0';
    }
});

// Drag and Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) {
        handleUpload(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleUpload(fileInput.files[0]);
    }
});

async function handleUpload(file) {
    // Show Optimistic UI
    const uploadItem = document.createElement('div');
    uploadItem.className = 'upload-item processing';
    uploadItem.innerHTML = `
        <div class="file-info">
            <span class="file-name">${file.name}</span>
            <span class="status-text">Uploading...</span>
        </div>
        <div class="progress-bar"><div class="fill" style="width: 0%"></div></div>
    `;

    // Clear empty state if needed
    const emptyState = uploadList.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    uploadList.prepend(uploadItem);
    modal.classList.remove('active'); // Close modal
    modal.style.opacity = '0';
    modal.style.pointerEvents = 'none';

    // FormData
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadItem.querySelector('.status-text').textContent = 'Processing...';
            uploadItem.querySelector('.fill').style.width = '50%';
            pollTaskStatus(data.task_id, uploadItem);
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (error) {
        console.error(error);
        uploadItem.querySelector('.status-text').textContent = 'Failed';
        uploadItem.querySelector('.status-text').style.color = 'var(--danger)';
    }
}

async function pollTaskStatus(taskId, element) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`http://localhost:8000/tasks/${taskId}`);
            const task = await response.json();

            if (task.status === 'completed') {
                clearInterval(interval);
                element.querySelector('.status-text').textContent = 'Completed';
                element.querySelector('.fill').style.width = '100%';
                element.querySelector('.fill').style.backgroundColor = 'var(--accent-green)';

                // Update Global Count (if applicable logic exists)
                // For now, we just display the result in the item
                const countSpan = document.createElement('span');
                countSpan.className = 'result-count';
                countSpan.textContent = ` Count: ${task.count}`;
                countSpan.style.color = 'var(--accent-gold)';
                countSpan.style.fontWeight = 'bold';
                countSpan.style.marginLeft = '10px';
                element.querySelector('.file-info').appendChild(countSpan);

                // Add Video Player / Link
                const videoContainer = document.createElement('div');
                videoContainer.className = 'result-video-container';
                videoContainer.style.marginTop = '10px';
                videoContainer.innerHTML = `
                    <video controls src="http://localhost:8000${task.video_url}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);"></video>
                    <a href="http://localhost:8000${task.video_url}" download class="download-link" style="display: block; margin-top: 5px; color: var(--accent-green); font-size: 0.8rem;">Download Processed Video</a>
                `;
                element.appendChild(videoContainer);

            } else if (task.status === 'failed') {
                clearInterval(interval);
                element.querySelector('.status-text').textContent = 'Failed';
                element.querySelector('.fill').style.backgroundColor = 'var(--danger)';
            }
        } catch (e) {
            console.error(e);
            clearInterval(interval);
        }
    }, 2000); // Poll every 2 seconds
}

// WebSocket Logic
// Connect to backend WebSocket (backend runs on port 8000)
const wsUrl = 'ws://localhost:8000/ws';
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    document.querySelector('.status-indicator').classList.add('connected');
    console.log("WebSocket connected");
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.count !== undefined) {
        updateCount(data.count);
    }
};

socket.onclose = () => {
    document.querySelector('.status-indicator').classList.remove('connected');
    console.log("WebSocket disconnected");
};

function updateCount(newCount) {
    const countElement = document.getElementById('current-count');

    // Only animate if count increased
    if (parseInt(countElement.textContent) < newCount) {
        // Trigger Animation
        countElement.classList.remove('pulse-animation');
        void countElement.offsetWidth; // Trigger reflow
        countElement.classList.add('pulse-animation');
    }

    countElement.textContent = newCount;
}

// Global Stats Polling (Optional fallback)
// setInterval(async () => { ...
