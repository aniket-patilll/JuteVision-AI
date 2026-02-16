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

                // Extract filename from the element text or use a generic name if needed
                const fileName = element.querySelector('.file-name').textContent || "Video Upload";

                // Add Result Display (Image or Video)
                const resultContainer = document.createElement('div');
                resultContainer.className = 'result-media-container';
                resultContainer.style.marginTop = '10px';

                const mediaUrl = `http://localhost:8000${task.video_url}`; // Backend sends URL in video_url field for both

                if (task.is_image) {
                    resultContainer.innerHTML = `
                        <img src="${mediaUrl}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">
                        <div class="result-actions" style="display: flex; gap: 10px; margin-top: 10px;">
                            <button class="btn-primary download-media-btn" data-url="${mediaUrl}" data-filename="detected_${fileName}.jpg" style="flex: 1; text-align: center; text-decoration: none; font-size: 0.9rem;">Download Image</button>
                            <button class="btn-primary view-analytics-btn" data-filename="${fileName}" style="flex: 1; font-size: 0.9rem;">View Analytics</button>
                        </div>
                    `;
                } else {
                    resultContainer.innerHTML = `
                        <video controls src="${mediaUrl}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);"></video>
                        <div class="result-actions" style="display: flex; gap: 10px; margin-top: 10px;">
                            <button class="btn-primary download-media-btn" data-url="${mediaUrl}" data-filename="detected_${fileName}.mp4" style="flex: 1; text-align: center; text-decoration: none; font-size: 0.9rem;">Download Video</button>
                            <button class="btn-primary view-analytics-btn" data-filename="${fileName}" style="flex: 1; font-size: 0.9rem;">View Analytics</button>
                        </div>
                    `;
                }

                element.appendChild(resultContainer);

                // Add Event Listeners
                const downloadBtn = resultContainer.querySelector('.download-media-btn');
                if (downloadBtn) {
                    downloadBtn.addEventListener('click', async () => {
                        const url = downloadBtn.getAttribute('data-url');
                        const filename = downloadBtn.getAttribute('data-filename');
                        try {
                            const response = await fetch(url);
                            const blob = await response.blob();
                            const blobUrl = window.URL.createObjectURL(blob);
                            const link = document.createElement('a');
                            link.href = blobUrl;
                            link.download = filename;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            window.URL.revokeObjectURL(blobUrl);
                        } catch (error) {
                            console.error('Download failed:', error);
                        }
                    });
                }

                const viewBtn = resultContainer.querySelector('.view-analytics-btn');
                if (viewBtn) {
                    viewBtn.addEventListener('click', () => {
                        localStorage.setItem('selectedAnalyticsFilter', fileName);
                        window.location.href = 'analytics.html';
                    });
                }

                // Add to Analytics Table
                addAnalyticsRow(fileName, task.count, "Completed");

                // Persist successful upload for dashboard view
                saveRecentUpload({
                    fileName,
                    count: task.count,
                    mediaUrl,
                    isImage: task.is_image,
                    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                });

            } else if (task.status === 'failed') {
                clearInterval(interval);
                element.querySelector('.status-text').textContent = 'Failed';
                element.querySelector('.fill').style.backgroundColor = 'var(--danger)';

                if (task.error) {
                    const errorSpan = document.createElement('div');
                    errorSpan.className = 'error-message';
                    errorSpan.textContent = task.error;
                    errorSpan.style.color = 'var(--danger)';
                    errorSpan.style.fontSize = '0.8rem';
                    errorSpan.style.marginTop = '4px';
                    element.querySelector('.file-info').appendChild(errorSpan);
                }
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

// Camera Toggle Logic
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM Loaded - Initializing Camera Toggle and Loading Results");

    // Load persisted dashboard items
    loadRecentUploads();

    const cameraToggle = document.getElementById('camera-toggle');
    const cameraFeed = document.getElementById('camera-feed');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const streamUrl = 'http://localhost:8000/stream';

    if (cameraToggle && cameraFeed) {
        // Function to update UI based on toggle state
        const updateCameraState = () => {
            console.log("Updating Camera State. Checked:", cameraToggle.checked);
            if (cameraToggle.checked) {
                // Enable Camera
                // Add timestamp to prevent caching issues when re-enabling
                cameraFeed.src = `${streamUrl}?t=${new Date().getTime()}`;
                cameraFeed.style.display = 'block';

                // Hide placeholder
                if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
            } else {
                // Disable Camera
                cameraFeed.style.display = 'none';
                cameraFeed.src = "";
                cameraFeed.removeAttribute('src'); // Stop stream

                // Show placeholder
                if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex';
            }
        };

        // Initialize state
        cameraToggle.checked = false;
        updateCameraState();

        // Event Listener
        cameraToggle.addEventListener('change', updateCameraState);
    } else {
        console.error("Camera elements not found in DOM");
    }

    // --- MODULAR MODE SWITCHING (v5) ---
    const modeRadios = document.querySelectorAll('input[name="analysis-mode"]');
    const totalBagsCard = document.querySelector('.total-count');
    const zoneStatsContainer = document.getElementById('zone-stats-container');

    function updateModeUI() {
        const selectedMode = document.querySelector('input[name="analysis-mode"]:checked').value;
        console.log("Mode switched to:", selectedMode);

        if (selectedMode === 'zone') {
            if (totalBagsCard) totalBagsCard.style.display = 'none';
            if (zoneStatsContainer) zoneStatsContainer.style.display = 'block';
        } else {
            if (totalBagsCard) totalBagsCard.style.display = 'block';
            if (zoneStatsContainer) zoneStatsContainer.style.display = 'none';
        }
    }

    modeRadios.forEach(radio => radio.addEventListener('change', updateModeUI));
    updateModeUI(); // Initial check
});


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
    }
    else if (data.type === "frame") {
        // LIVE PROCESSING FEEDBACK
        const cameraFeed = document.getElementById('camera-feed');
        const cameraPlaceholder = document.getElementById('camera-placeholder');

        if (cameraFeed && cameraPlaceholder) {
            cameraFeed.src = `data:image/jpeg;base64,${data.data}`;
            cameraFeed.style.display = 'block';
            cameraPlaceholder.style.display = 'none';
        }

        // 'data.count' in a frame message is the LIVE ROI Occupancy (Sacks in ROI)
        if (data.count !== undefined) {
            updateROIStatus(data.count);
        }
    }
    // 'data.count' in a global message is the cumulative session total
    else if (data.count !== undefined) {
        updateTotalCount(data.count);
    }
};

socket.onclose = () => {
    document.querySelector('.status-indicator').classList.remove('connected');
    console.log("WebSocket disconnected");
};

// Update the "Total Bags" card (Session Cumulative)
function updateTotalCount(newCount) {
    const countElement = document.getElementById('current-count');
    if (!countElement) return;

    const currentTotal = parseInt(countElement.textContent) || 0;

    // Only animate if count increased
    if (currentTotal < newCount) {
        countElement.classList.remove('pulse-animation');
        void countElement.offsetWidth; // Trigger reflow
        countElement.classList.add('pulse-animation');
    }

    countElement.textContent = newCount;
    localStorage.setItem('currentTotalBags', newCount);
}

// Update the "Status In ROI" card (Live Occupancy)
function updateROIStatus(occupancy) {
    const insideCountElement = document.getElementById('inside-count');
    if (insideCountElement) {
        insideCountElement.textContent = occupancy;
    }
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
const exportBtn = document.getElementById('export-btn');
if (exportBtn) {
    exportBtn.addEventListener('click', () => {
        // Get data from localStorage
        const storedData = localStorage.getItem('analyticsData');
        if (!storedData) {
            alert("No data to export");
            return;
        }

        const analyticsData = JSON.parse(storedData);
        if (analyticsData.length === 0) {
            alert("No data to export");
            return;
        }

        // CSV Header
        let csvContent = "data:text/csv;charset=utf-8,Time,File,Count,Status\n";

        analyticsData.forEach(item => {
            csvContent += `${item.time},${item.filename},${item.count},${item.status}\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "cctv_visioncount_analytics.csv");
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

    // Reset Zone Stats (Modular)
    const insideCount = document.getElementById('inside-count');
    if (insideCount) insideCount.textContent = '0';

    // Clear Analytics Table
    const tableBody = document.getElementById('analytics-table-body');
    if (tableBody) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No recent activity</td>
            </tr>
        `;
    }

    // Reset Metrics
    const metricUploads = document.getElementById('metric-uploads');
    const metricAvg = document.getElementById('metric-avg');
    const metricSuccess = document.getElementById('metric-success');

    if (metricUploads) metricUploads.textContent = '0';
    if (metricAvg) metricAvg.textContent = '0';
    if (metricSuccess) metricSuccess.textContent = '100%';

    // Clear Upload List
    const uploadList = document.getElementById('upload-list');
    if (uploadList) {
        uploadList.innerHTML = '<div class="empty-state">No active uploads</div>';
    }

    // Clear LocalStorage
    localStorage.removeItem('analyticsData');
    localStorage.removeItem('recentUploads');
    localStorage.removeItem('currentTotalBags'); // Also clear count persistence if we add it
}

// Result Persistence Helpers
function saveRecentUpload(item) {
    let recent = JSON.parse(localStorage.getItem('recentUploads') || '[]');
    recent.unshift(item);
    if (recent.length > 5) recent = recent.slice(0, 5); // Keep only last 5 for dashboard
    localStorage.setItem('recentUploads', JSON.stringify(recent));

    // Also update current total bags persistence
    const currentTotal = parseInt(localStorage.getItem('currentTotalBags') || '0');
    localStorage.setItem('currentTotalBags', currentTotal + item.count);
}

function loadRecentUploads() {
    const recent = JSON.parse(localStorage.getItem('recentUploads') || '[]');
    const currentTotal = localStorage.getItem('currentTotalBags') || '0';

    // Restore count
    updateCount(parseInt(currentTotal));

    if (recent.length > 0) {
        const uploadList = document.getElementById('upload-list');
        const emptyState = uploadList.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        recent.forEach(item => {
            const uploadItem = document.createElement('div');
            uploadItem.className = 'upload-item completed'; // It's already completed

            const mediaHtml = item.isImage
                ? `<img src="${item.mediaUrl}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">`
                : `<video controls src="${item.mediaUrl}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);"></video>`;

            uploadItem.innerHTML = `
                <div class="file-info">
                    <span class="file-name">${item.fileName}</span>
                    <span class="status-text" style="color: var(--accent-green)">Completed</span>
                    <span class="result-count" style="color: var(--accent-gold); font-weight: bold; margin-left: 10px;">Count: ${item.count}</span>
                </div>
                <div class="progress-bar"><div class="fill" style="width: 100%; background-color: var(--accent-green)"></div></div>
                <div class="result-media-container" style="marginTop: 10px;">
                    ${mediaHtml}
                    <div class="result-actions" style="display: flex; gap: 10px; margin-top: 10px;">
                        <button class="btn-primary download-media-btn" data-url="${item.mediaUrl}" data-filename="detected_${item.fileName}${item.isImage ? '.jpg' : '.mp4'}" style="flex: 1; text-align: center; text-decoration: none; font-size: 0.9rem;">Download ${item.isImage ? 'Image' : 'Video'}</button>
                        <button class="btn-primary view-analytics-btn" data-filename="${item.fileName}" style="flex: 1; font-size: 0.9rem;">View Analytics</button>
                    </div>
                </div>
            `;

            uploadList.appendChild(uploadItem);

            // Re-attach listeners
            const downloadBtn = uploadItem.querySelector('.download-media-btn');
            downloadBtn.addEventListener('click', async () => {
                const url = downloadBtn.getAttribute('data-url');
                const filename = downloadBtn.getAttribute('data-filename');
                try {
                    const response = await fetch(url);
                    const blob = await response.blob();
                    const blobUrl = window.URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = blobUrl;
                    link.download = filename;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    window.URL.revokeObjectURL(blobUrl);
                } catch (error) {
                    console.error('Download failed:', error);
                }
            });

            const viewBtn = uploadItem.querySelector('.view-analytics-btn');
            viewBtn.addEventListener('click', () => {
                localStorage.setItem('selectedAnalyticsFilter', item.fileName);
                window.location.href = 'analytics.html';
            });
        });
    }
}
// --- ENHANCED DOWNLOAD LOGIC (v10.2) ---
document.addEventListener('DOMContentLoaded', () => {
    const downloadBtn = document.getElementById('download-sample-btn');
    const dropdownMenu = document.getElementById('download-dropdown');
    const sampleModal = document.getElementById('sample-media-modal');
    const closeSampleModal = document.getElementById('close-sample-modal');
    const modalTitle = document.getElementById('modal-mode-title');
    const sampleGrid = document.getElementById('sample-grid');

    // Sample Data Store
    const sampleData = {
        static: [
            { name: "01_Jute_Stack_Truck", type: "image", url: "assets/samples/static_1.jpg", thumb: "assets/samples/static_1.jpg" },
            { name: "02_Jute_Bags_CloseUp", type: "image", url: "assets/samples/static_2.jpg", thumb: "assets/samples/static_2.jpg" },
            { name: "03_Jute_Warehouse_Grid", type: "image", url: "assets/samples/static_3.jpg", thumb: "assets/samples/static_3.jpg" },
            { name: "04_Jute_Loading_Bay", type: "image", url: "assets/samples/static_4.jpg", thumb: "assets/samples/static_4.jpg" }
        ],
        scanning: [
            { name: "Scanning Mode 01", type: "video", url: "assets/samples/scanning_1.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Scanning Mode 02", type: "video", url: "assets/samples/scanning_2.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Scanning Mode 03", type: "video", url: "assets/samples/scanning_3.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Scanning Mode 04", type: "video", url: "assets/samples/scanning_4.mp4", thumb: "assets/samples/scanning_thumb.jpg" }
        ],
        zone: [
            { name: "Zone Mode 01", type: "video", url: "assets/samples/zone_1.mp4", thumb: "assets/samples/zone_thumb.jpg" },
            { name: "Zone Mode 02", type: "video", url: "assets/samples/zone_2.mp4", thumb: "assets/samples/zone_thumb.jpg" }
        ]
    };

    // Toggle Dropdown
    if (downloadBtn && dropdownMenu) {
        downloadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropdownMenu.classList.toggle('show');
        });

        // Close dropdown when clicking outside
        window.addEventListener('click', () => {
            dropdownMenu.classList.remove('show');
        });
    }

    // Handle Dropdown Item Clicks
    const dropdownItems = document.querySelectorAll('.dropdown-item');
    dropdownItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const mode = item.getAttribute('data-mode');
            const modeTitle = item.textContent.trim();

            if (mode === 'static' || mode === 'scanning' || mode === 'zone') {
                downloadModeZip(mode);
            } else {
                openSampleModal(mode, modeTitle);
            }
        });
    });

    async function downloadModeZip(mode) {
        const items = sampleData[mode] || [];
        if (items.length === 0) return;

        const zip = new JSZip();
        const zipName = `jute_${mode}_samples_bundle.zip`;
        const folder = zip.folder(`jute_${mode}_samples`);

        const fetchPromises = items.map(async (sample) => {
            try {
                const response = await fetch(sample.url);
                const blob = await response.blob();
                const extension = sample.type === 'video' ? '.mp4' : '.jpg';
                const fileName = sample.name.replace(/ /g, '_') + extension;
                folder.file(fileName, blob);
            } catch (err) {
                console.error(`Failed to fetch ${sample.name}:`, err);
            }
        });

        await Promise.all(fetchPromises);

        const content = await zip.generateAsync({ type: "blob" });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = zipName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setTimeout(() => URL.revokeObjectURL(link.href), 100);
    }

    function openSampleModal(mode, title) {
        modalTitle.textContent = title;
        sampleGrid.innerHTML = ''; // Clear previous

        const items = sampleData[mode] || [];

        if (items.length === 0) {
            sampleGrid.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); padding: 2rem;">No samples available for this mode yet.</p>`;
        } else {
            items.forEach(sample => {
                const card = document.createElement('div');
                card.className = 'sample-card';
                card.innerHTML = `
                    ${sample.type === 'video'
                        ? `<video src="${sample.url}" class="sample-preview" muted onmouseover="this.play()" onmouseout="this.pause(); this.currentTime=0;"></video>`
                        : `<img src="${sample.url}" class="sample-preview" alt="${sample.name}">`}
                    <div class="sample-info">
                        <span class="sample-name" title="${sample.name}">${sample.name}</span>
                        <button class="btn-sample-download" data-url="${sample.url}" data-name="${sample.name}">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                            Download ${sample.type === 'video' ? 'Video' : 'Image'}
                        </button>
                    </div>
                `;
                sampleGrid.appendChild(card);

                // Add Download Logic
                card.querySelector('.btn-sample-download').addEventListener('click', () => {
                    const link = document.createElement('a');
                    link.href = sample.url;
                    link.download = `${sample.name.toLowerCase().replace(/ /g, '_')}.${sample.type === 'video' ? 'mp4' : 'jpg'}`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                });
            });
        }

        sampleModal.classList.add('active');
        sampleModal.style.opacity = '1';
        sampleModal.style.pointerEvents = 'auto';
    }

    // Close Modal Logic
    if (closeSampleModal) {
        closeSampleModal.addEventListener('click', () => {
            sampleModal.classList.remove('active');
            sampleModal.style.opacity = '0';
            sampleModal.style.pointerEvents = 'none';
        });
    }

    window.addEventListener('click', (e) => {
        if (e.target === sampleModal) {
            sampleModal.classList.remove('active');
            sampleModal.style.opacity = '0';
            sampleModal.style.pointerEvents = 'none';
        }
    });
});
