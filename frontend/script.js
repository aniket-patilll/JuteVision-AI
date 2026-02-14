import { requireAuth, signOut } from './auth.js';

// Protect Route
requireAuth();

const uploadBtn = document.getElementById('upload-btn');
const modal = document.getElementById('upload-modal');
const closeModal = document.getElementById('close-modal');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadList = document.getElementById('upload-list');
const currentCount = document.getElementById('current-count');

// Logout Logic
const logoutBtn = document.getElementById('nav-logout');
if (logoutBtn) {
    logoutBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        await signOut();
        window.location.href = 'login.html';
    });
}

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

    // Get Selected Mode
    const selectedMode = document.querySelector('input[name="analysis-mode"]:checked').value;
    formData.append('mode', selectedMode);

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

                // Add to Analytics Table
                // Extract filename from the element text or use a generic name if needed
                const fileName = element.querySelector('.file-name').textContent || "Video Upload";
                addAnalyticsRow(fileName, task.count, "Completed");

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

function addAnalyticsRow(filename, count, status) {
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Get existing data from localStorage
    let analyticsData = [];
    const storedData = localStorage.getItem('analyticsData');
    if (storedData) {
        analyticsData = JSON.parse(storedData);
    }

    // Add new entry
    analyticsData.unshift({
        time: timeString,
        filename: filename,
        count: count,
        status: status
    });

    // Keep only last 50 entries
    if (analyticsData.length > 50) {
        analyticsData = analyticsData.slice(0, 50);
    }

    // Save back to localStorage
    localStorage.setItem('analyticsData', JSON.stringify(analyticsData));
}

// WebSocket Connection Logic
// Connect to backend WebSocket (backend runs on port 8000)
const wsUrl = 'ws://localhost:8000/ws';
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    document.querySelector('.status-indicator').classList.add('connected');
    console.log("WebSocket connected");
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.event === "reset") {
        resetUI();
    } else if (data.count !== undefined) {
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


// Navigation Logic
const navDashboard = document.getElementById('nav-dashboard');
const navAnalytics = document.getElementById('nav-analytics');
const analyticsSection = document.getElementById('analytics-section');


// Update details
navDashboard.addEventListener('click', (e) => {
    e.preventDefault();
    navDashboard.classList.add('active');
    navAnalytics.classList.remove('active');
    document.querySelector('.main-content').scrollTo({ top: 0, behavior: 'smooth' });
});



// Update Stats Logic
function updateGlobalStats() {
    const tableBody = document.getElementById('analytics-table-body');
    const rows = Array.from(tableBody.querySelectorAll('tr'));

    // Filter out "No recent activity" row
    const dataRows = rows.filter(row => row.cells.length > 1);

    const totalUploads = dataRows.length;
    let totalBags = 0;
    let successCount = 0;

    dataRows.forEach(row => {
        const count = parseInt(row.cells[2].textContent) || 0;
        const status = row.cells[3].textContent.trim();

        totalBags += count;
        if (status.includes('Completed') || status.includes('Verified')) {
            successCount++;
        }
    });

    const avgBags = totalUploads > 0 ? Math.round(totalBags / totalUploads) : 0;
    const successRate = totalUploads > 0 ? Math.round((successCount / totalUploads) * 100) : 100;

    document.getElementById('metric-uploads').textContent = totalUploads;
    document.getElementById('metric-avg').textContent = avgBags;
    document.getElementById('metric-success').textContent = `${successRate}%`;
}

// Export Data
const exportBtn = document.querySelector('.btn-secondary'); // "Export Data" button
if (exportBtn) {
    exportBtn.addEventListener('click', () => {
        const tableBody = document.getElementById('analytics-table-body');
        const rows = Array.from(tableBody.querySelectorAll('tr'));

        // CSV Header
        let csvContent = "data:text/csv;charset=utf-8,Time,File,Count,Status\n";

        rows.forEach(row => {
            // Skip empty state
            if (row.cells.length <= 1) return;

            const cols = Array.from(row.querySelectorAll('td'))
                .map(td => td.textContent.trim())
                .join(",");
            csvContent += cols + "\r\n";
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "jutevision_analytics.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

// Reset Session Logic
const resetBtn = document.getElementById('reset-btn');
if (resetBtn) {
    resetBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to reset the session? This will clear all counts and history.')) {
            try {
                const response = await fetch('http://localhost:8000/reset', { method: 'POST' });
                if (response.ok) {
                    // Clear UI
                    resetUI();
                }
            } catch (error) {
                console.error('Failed to reset:', error);
            }
        }
    });
}

function resetUI() {
    // Reset Total Count
    updateCount(0);

    // Clear Analytics Table
    const tableBody = document.getElementById('analytics-table-body');
    tableBody.innerHTML = `
        <tr>
            <td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No recent activity</td>
        </tr>
    `;

    // Reset Metrics
    document.getElementById('metric-uploads').textContent = '0';
    document.getElementById('metric-avg').textContent = '0';
    document.getElementById('metric-success').textContent = '100%';

    // Clear Upload List
    const uploadList = document.getElementById('upload-list');
    uploadList.innerHTML = '<div class="empty-state">No active uploads</div>';
}

