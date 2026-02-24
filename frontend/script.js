import { requireAuth, signOut } from './auth.js';
import { API_BASE_URL, ENDPOINTS, getApiUrl, getWsUrl } from './config.js';

// Protect Route & Get User ID
let userId = null;
const initAuth = async () => {
    const session = await requireAuth();
    if (session) {
        userId = session.user.id;
        connectWebSocket(); // Start user-specific dynamic updates
        loadRecentUploads();
        updateGlobalStats();
    }
};
initAuth();

const getAnalyticsKey = () => userId ? `analyticsData_${userId}` : 'analyticsData';

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

// Toggle Volume Depth Input and Filter Upload Types
const modeRadios = document.querySelectorAll('input[name="analysis-mode"]');
const depthContainer = document.getElementById('volume-depth-container');
const mainFileInput = document.getElementById('file-input');
const uploadSupportText = document.getElementById('upload-support-text'); // Need to add this ID to HTML

modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        const mode = e.target.value;

        // Depth Container
        if (mode === 'volume') {
            depthContainer.style.display = 'block';
        } else {
            depthContainer.style.display = 'none';
        }

        // Dynamic File Type Restrictions
        if (mode === 'static') {
            mainFileInput.accept = 'image/jpeg,image/png,image/jpg,image/webp';
            if (uploadSupportText) uploadSupportText.textContent = 'Supports Images Only (JPG, PNG)';
        } else if (mode === 'volume') {
            mainFileInput.accept = 'video/mp4,video/avi,image/jpeg,image/png,image/jpg,image/webp';
            if (uploadSupportText) uploadSupportText.textContent = 'Supports Videos (MP4) & Images (JPG)';
        } else {
            // Zone, Scanning, Godown
            mainFileInput.accept = 'video/mp4,video/avi,video/mov';
            if (uploadSupportText) uploadSupportText.textContent = 'Supports Videos Only (MP4, AVI)';
        }
    });
});

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

    // Add manual depth override if applicable
    if (selectedMode === 'volume') {
        const depthInput = document.getElementById('manual-depth-input');
        if (depthInput && depthInput.value) {
            formData.append('depth_override', depthInput.value);
        }
    }

    // v13.5 Isolation Fix: Explicitly send Supabase UID
    if (userId) {
        formData.append('user_id', userId);
    }

    try {
        const response = await fetch(getApiUrl(ENDPOINTS.UPLOAD), {
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
            const response = await fetch(`${API_BASE_URL}${ENDPOINTS.TASKS}/${taskId}`);
            const task = await response.json();

            if (task.status === 'completed') {
                clearInterval(interval);
                element.querySelector('.status-text').textContent = 'Completed';
                element.querySelector('.fill').style.width = '100%';
                element.querySelector('.fill').style.backgroundColor = 'var(--accent-green)';

                // Prevent duplicate appending if this runs twice before interval fully clears
                if (!element.querySelector('.result-count')) {
                    const countSpan = document.createElement('span');
                    countSpan.className = 'result-count';

                    if (task.estimation_mode) {
                        const depthLabel = task.depth_override_used ? "Known Depth" : "Est. Depth";
                        countSpan.innerHTML = `<br><span style="color:#00C853; font-size:0.9rem;">👁️ Visible: ${task.visible_count}</span> | <span style="color:#42A5F5; font-size:0.9rem;">🧊 ${depthLabel}: ${task.depth_layers} Layers</span> <br> <span style="color:var(--accent-gold); font-size:1.1rem; display:inline-block; margin-top:5px;">📦 Total Prediction: ${task.estimated_total} Sacks</span>`;
                        countSpan.style.display = 'block';
                        countSpan.style.marginTop = '5px';
                        countSpan.style.fontWeight = 'bold';
                    } else {
                        countSpan.textContent = ` Count: ${task.count}`;
                        countSpan.style.color = 'var(--accent-gold)';
                        countSpan.style.fontWeight = 'bold';
                        countSpan.style.marginLeft = '10px';
                    }

                    element.querySelector('.file-info').appendChild(countSpan);
                }

                // Extract filename from the element text or use a generic name if needed
                const fileName = element.querySelector('.file-name').textContent || "Video Upload";

                // Add Result Display (Image or Video)
                if (!element.querySelector('.result-media-container')) {
                    const resultContainer = document.createElement('div');
                    resultContainer.className = 'result-media-container';
                    resultContainer.style.marginTop = '10px';

                    const mediaUrl = `${API_BASE_URL}${task.video_url}`; // Backend sends URL in video_url field for both

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
                } // End of if(!element.querySelector('.result-media-container'))

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
    const storageKey = getAnalyticsKey();
    const storedData = localStorage.getItem(storageKey);
    if (storedData) {
        analyticsData = JSON.parse(storedData);
    }

    // Add new entry
    analyticsData.unshift({
        time: timeString,
        filename: filename,
        count: count,
        status: status,
        actualCount: count
    });

    // Keep only last 50 entries
    if (analyticsData.length > 50) {
        analyticsData = analyticsData.slice(0, 50);
    }

    // Save back to localStorage
    localStorage.setItem(getAnalyticsKey(), JSON.stringify(analyticsData));
}

// Camera Toggle Logic
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM Loaded - Initializing Camera Toggle and Loading Results");

    // Results will be loaded via initAuth() once userId is ready

    const cameraToggle = document.getElementById('camera-toggle');
    const cameraFeed = document.getElementById('camera-feed');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const streamUrl = getApiUrl(ENDPOINTS.STREAM);

    if (cameraToggle && cameraFeed) {
        // Function to update UI based on toggle state
        const updateCameraState = async () => {
            console.log("Updating Camera State. Checked:", cameraToggle.checked);
            if (cameraToggle.checked) {
                // Enable Camera - Inform backend first to power up hardware
                try {
                    await fetch(getApiUrl(ENDPOINTS.CAMERA_ON), { method: 'POST' });
                } catch (e) {
                    console.error("Hardware activation signal failed:", e);
                }

                // Add timestamp to prevent caching issues when re-enabling
                cameraFeed.src = `${streamUrl}?t=${new Date().getTime()}`;
                cameraFeed.style.display = 'block';

                // Hide placeholder
                if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
            } else {
                // Disable Camera - Inform backend to kill hardware stream immediately
                try {
                    await fetch(getApiUrl(ENDPOINTS.CAMERA_OFF), { method: 'POST' });
                } catch (e) {
                    console.error("Hardware deactivation signal failed:", e);
                }

                cameraFeed.style.display = 'none';

                // v15.5 Precision Fix: Force browser to drop MJPEG connection
                cameraFeed.src = "about:blank";
                cameraFeed.removeAttribute('src');

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

        const godownSection = document.getElementById('godown-section');
        const singleCamHeader = document.getElementById('single-cam-header');
        const multiCamHeader = document.getElementById('multi-cam-header');
        const singleCamView = document.getElementById('single-cam-view');
        const multiCctvGrid = document.getElementById('multi-cctv-grid');
        const singleCamControls = document.getElementById('single-cam-controls');

        // Reset all
        if (totalBagsCard) totalBagsCard.style.display = 'none';
        if (zoneStatsContainer) zoneStatsContainer.style.display = 'none';
        if (godownSection) godownSection.style.display = 'none';

        // Default: show single cam view
        if (singleCamHeader) singleCamHeader.style.display = '';
        if (multiCamHeader) multiCamHeader.style.display = 'none';
        if (singleCamView) singleCamView.style.display = '';
        if (multiCctvGrid) multiCctvGrid.style.display = 'none';
        if (singleCamControls) singleCamControls.style.display = '';

        if (selectedMode === 'zone' || selectedMode === 'conveyor') {
            if (zoneStatsContainer) zoneStatsContainer.style.display = 'block';
        } else if (selectedMode === 'multi-cctv') {
            // Switch Live Feed card to multi-camera grid
            if (singleCamHeader) singleCamHeader.style.display = 'none';
            if (multiCamHeader) multiCamHeader.style.display = 'flex';
            if (singleCamView) singleCamView.style.display = 'none';
            if (multiCctvGrid) multiCctvGrid.style.display = 'grid';
            if (singleCamControls) singleCamControls.style.display = 'none';

            // Auto-close the upload modal and scroll to the grid
            const uploadModal = document.getElementById('upload-modal');
            if (uploadModal && uploadModal.style.display !== 'none') {
                uploadModal.style.display = 'none';
                uploadModal.style.opacity = '0';
                uploadModal.style.pointerEvents = 'none';
            }
            setTimeout(() => {
                const feedCard = document.querySelector('.video-feed-card');
                if (feedCard) feedCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        } else if (selectedMode === 'godown') {
            if (godownSection) godownSection.style.display = 'block';
            loadGodownStatus();
        } else {
            if (totalBagsCard) totalBagsCard.style.display = 'block';
        }
    }

    modeRadios.forEach(radio => radio.addEventListener('change', updateModeUI));
    updateModeUI(); // Initial check

    // --- GODOWN CCTV START BUTTON ---
    const godownStartCctvBtn = document.getElementById('godown-start-cctv-btn');
    if (godownStartCctvBtn) {
        godownStartCctvBtn.addEventListener('click', async () => {
            // Close modal
            const uploadModal = document.getElementById('upload-modal');
            if (uploadModal) {
                uploadModal.classList.remove('active');
                uploadModal.style.opacity = '0';
                uploadModal.style.pointerEvents = 'none';
            }

            // Get line position from slider
            const slider = document.getElementById('godown-line-slider');
            const linePos = slider ? slider.value : 50;

            // Start godown live stream on backend
            try {
                const formData = new FormData();
                formData.append('line_position', linePos);
                if (userId) formData.append('user_id', userId);
                await fetch(getApiUrl(ENDPOINTS.GODOWN_START_LIVE), { method: 'POST', body: formData });
            } catch (e) {
                console.error('Failed to start godown live:', e);
            }

            // Switch feed to godown stream
            const cameraFeed = document.getElementById('camera-feed');
            const cameraPlaceholder = document.getElementById('camera-placeholder');
            if (cameraFeed) {
                cameraFeed.src = `${getApiUrl(ENDPOINTS.GODOWN_STREAM)}?t=${new Date().getTime()}`;
                cameraFeed.style.display = 'block';
            }
            if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';

            // Show godown section
            const godownSection = document.getElementById('godown-section');
            if (godownSection) godownSection.style.display = 'block';
            loadGodownStatus();
        });
    }
});


// WebSocket Connection Logic
// Connect to backend WebSocket (backend runs on port 8000)
// wait for userId before connecting
let socket = null;
const connectWebSocket = () => {
    if (!userId) return;
    const wsUrl = `${getWsUrl(ENDPOINTS.WS)}/${userId}`;
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        document.querySelector('.status-indicator').classList.add('connected');
        console.log("WebSocket connected for user:", userId);
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("WS Status Update:", data);

        if (data.event === "reset") {
            resetUI();
        }
        else if (data.type === "multi_cctv_frame") {
            // Multi-CCTV frame update
            const camId = data.camera_id;
            const cellBody = document.querySelector(`#cam-body-${camId}`);
            const cellCount = document.querySelector(`#cam-count-${camId}`);
            if (cellBody) {
                let img = cellBody.querySelector('img');
                if (!img) {
                    const waitText = cellBody.querySelector('.cam-waiting-text');
                    if (waitText) waitText.remove();
                    img = document.createElement('img');
                    img.style.cssText = 'width:100%;height:100%;object-fit:cover;position:absolute;top:0;left:0;';
                    cellBody.insertBefore(img, cellBody.firstChild);
                }
                img.src = `data:image/jpeg;base64,${data.data}`;
            }
            if (cellCount) cellCount.textContent = `${data.count} sacks`;
            updateMultiCctvTotal();
        }
        else if (data.event === "godown_in" || data.event === "godown_out") {
            // Godown real-time event
            updateGodownStats(data);
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
        console.log("WebSocket disconnected. Retrying...");
        setTimeout(connectWebSocket, 3000);
    };
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
    localStorage.setItem(userId ? `currentTotalBags_${userId}` : 'currentTotalBags', newCount);
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
    if (!tableBody) return; // Not on dashboard page — skip
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
    const successRate = totalUploads > 0 ? Math.round((successCount / totalUploads) * 100) : 0;

    document.getElementById('metric-uploads').textContent = totalUploads;
    document.getElementById('metric-avg').textContent = avgBags;
    document.getElementById('metric-success').textContent = `${successRate}%`;
}

// Export Data
const exportBtn = document.getElementById('export-btn');
if (exportBtn) {
    exportBtn.addEventListener('click', () => {
        // Get data from localStorage
        const storedData = localStorage.getItem(getAnalyticsKey());
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
                // Send user_id so the backend scopes the reset to THIS user only
                const resetUrl = `${getApiUrl(ENDPOINTS.RESET)}?user_id=${userId || 'anonymous'}`;
                const response = await fetch(resetUrl, { method: 'POST' });
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
    updateTotalCount(0);

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
    // Clear LocalStorage for this user
    localStorage.removeItem(getAnalyticsKey());
    localStorage.removeItem(userId ? `recentUploads_${userId}` : 'recentUploads');
    localStorage.removeItem(userId ? `currentTotalBags_${userId}` : 'currentTotalBags');
}

// Result Persistence Helpers
function saveRecentUpload(item) {
    const recentKey = userId ? `recentUploads_${userId}` : 'recentUploads';
    const totalKey = userId ? `currentTotalBags_${userId}` : 'currentTotalBags';

    let recent = JSON.parse(localStorage.getItem(recentKey) || '[]');
    recent.unshift(item);
    if (recent.length > 5) recent = recent.slice(0, 5); // Keep only last 5 for dashboard
    localStorage.setItem(recentKey, JSON.stringify(recent));

    // Also update current total bags persistence
    const currentTotal = parseInt(localStorage.getItem(totalKey) || '0');
    localStorage.setItem(totalKey, currentTotal + item.count);
}

function loadRecentUploads() {
    const recentKey = userId ? `recentUploads_${userId}` : 'recentUploads';
    const totalKey = userId ? `currentTotalBags_${userId}` : 'currentTotalBags';

    const recent = JSON.parse(localStorage.getItem(recentKey) || '[]');
    const currentTotal = localStorage.getItem(totalKey) || '0';

    // Restore count
    updateTotalCount(parseInt(currentTotal));

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
        conveyor: [
            { name: "Godown Conveyor Belt Sample", type: "video", url: "assets/samples/conveyor_1.mp4", thumb: "assets/samples/conveyor_thumb.jpg" }
        ],
        static: [
            { name: "01_Jute_Stack_Truck", type: "image", url: "assets/samples/static_1.jpg", thumb: "assets/samples/static_1.jpg" },
            { name: "03_Jute_Warehouse_Grid", type: "image", url: "assets/samples/static_3.jpg", thumb: "assets/samples/static_3.jpg" },
            { name: "05_Shared_Image", type: "image", url: "assets/samples/static_5.jpg", thumb: "assets/samples/static_5.jpg" },
            { name: "06_Shared_Image_3", type: "image", url: "assets/samples/static_6.jpg", thumb: "assets/samples/static_6.jpg" }
        ],
        scanning: [
            { name: "Scanning Mode 01", type: "video", url: "assets/samples/scanning_1.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Scanning Mode 02", type: "video", url: "assets/samples/scanning_2.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Scanning Mode 03", type: "video", url: "assets/samples/scanning_3.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Scanning Mode 04", type: "video", url: "assets/samples/scanning_4.mp4", thumb: "assets/samples/scanning_thumb.jpg" }
        ],
        zone: [
            { name: "Zone Mode 02", type: "video", url: "assets/samples/zone_2.mp4", thumb: "assets/samples/zone_thumb.jpg" }
        ],
        volume: [
            { name: "Volume Estimation 01", type: "image", url: "assets/samples/Volume_estimation_01.jpeg", thumb: "assets/samples/static_1.jpg" },
            { name: "Volume Estimation 02", type: "image", url: "assets/samples/Volume_estimation_02.jpeg", thumb: "assets/samples/static_1.jpg" },
            { name: "Volume Estimation 03", type: "video", url: "assets/samples/Volume_estimation_03.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "Volume Estimation 04", type: "image", url: "assets/samples/Volume_estmation_04.jpeg", thumb: "assets/samples/static_1.jpg" },
            { name: "Volume Estimation 05", type: "image", url: "assets/samples/Volume_estimation_05.jpeg", thumb: "assets/samples/static_1.jpg" }
        ],
        multicctv: [
            { name: "CCTV Angle 1", type: "video", url: "assets/samples/scanning_mode_1.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "CCTV Angle 2", type: "video", url: "assets/samples/scanning_mode_2.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "CCTV Angle 3", type: "video", url: "assets/samples/scanning_mode_3.mp4", thumb: "assets/samples/scanning_thumb.jpg" },
            { name: "CCTV Angle 4", type: "video", url: "assets/samples/scanning_mode_4.mp4", thumb: "assets/samples/scanning_thumb.jpg" }
        ],
        godown: [
            { name: "Godown Sample Video", type: "video", url: "assets/samples/godown_sample.mp4", thumb: "assets/samples/conveyor_thumb.jpg" },
            { name: "Godown Video 02", type: "video", url: "assets/samples/Godown_mode02.mp4", thumb: "assets/samples/conveyor_thumb.jpg" }
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

            if (sampleData[mode]) {
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

// ============================================================
// MULTI-CCTV MODE LOGIC
// ============================================================

let multiCctvCameras = {};

document.addEventListener('DOMContentLoaded', () => {
    const addCameraBtn = document.getElementById('add-camera-btn');
    if (addCameraBtn) {
        addCameraBtn.addEventListener('click', addCameraCell);
    }

    // Single Upload button — shows grid layout selection
    const multiUploadBtn = document.getElementById('multi-upload-btn');
    if (multiUploadBtn) {
        multiUploadBtn.addEventListener('click', () => {
            showGridUploadDialog();
        });
    }

    // Single Live button — starts webcam on next idle camera
    const multiLiveBtn = document.getElementById('multi-live-btn');
    if (multiLiveBtn) {
        multiLiveBtn.addEventListener('click', () => {
            const idleCam = findIdleCamera();
            if (!idleCam) {
                alert('No idle camera available. Add a new camera first.');
                return;
            }
            startLiveCamera(idleCam);
        });
    }

    // Godown slider
    const godownSlider = document.getElementById('godown-line-slider');
    let godownSliderTimeout;
    if (godownSlider) {
        godownSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            const label = document.getElementById('godown-line-value');
            if (label) label.textContent = `Position: ${val}%`;

            // Debounce API call to avoid spamming the backend while dragging
            clearTimeout(godownSliderTimeout);
            godownSliderTimeout = setTimeout(async () => {
                try {
                    const formData = new FormData();
                    formData.append('line_position', val);
                    await fetch(getApiUrl(ENDPOINTS.GODOWN_UPDATE_LINE), { method: 'POST', body: formData });
                } catch (err) {
                    console.error('Failed to update godown line:', err);
                }
            }, 100);
        });
    }

    // Godown set baseline
    const baselineBtn = document.getElementById('godown-set-baseline-btn');
    if (baselineBtn) {
        baselineBtn.addEventListener('click', async () => {
            const count = prompt('Enter current inventory count (baseline):');
            if (count !== null && !isNaN(count)) {
                const formData = new FormData();
                formData.append('count', parseInt(count));
                await fetch(getApiUrl(ENDPOINTS.GODOWN_BASELINE), { method: 'POST', body: formData });
                loadGodownStatus();
            }
        });
    }

    // Godown reset daily
    const resetDailyBtn = document.getElementById('godown-reset-daily-btn');
    if (resetDailyBtn) {
        resetDailyBtn.addEventListener('click', async () => {
            if (confirm('Reset daily in/out counters?')) {
                await fetch(getApiUrl(ENDPOINTS.GODOWN_RESET_DAILY), { method: 'POST' });
                loadGodownStatus();
            }
        });
    }
});

let multiCctvCameraCounter = 0;

// ---- Grid Upload Dialog ----

function showGridUploadDialog() {
    // Remove existing dialog if any
    const existing = document.getElementById('grid-upload-dialog');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'grid-upload-dialog';
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(4px);';
    overlay.innerHTML = `
        <div style="background:white;border-radius:16px;padding:2rem;max-width:480px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
            <h3 style="margin:0 0 0.5rem;color:#1a1a1a;font-size:1.1rem;">Upload Videos to Multi-Camera Grid</h3>
            <p style="color:#64748b;font-size:0.85rem;margin:0 0 1.25rem;">Select a layout, then pick individual video files for each camera:</p>

            <!-- Preset grid options -->
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-bottom:1rem;">
                <button class="grid-option-btn" data-rows="1" data-cols="1" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">1</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">Single</div>
                </button>
                <button class="grid-option-btn" data-rows="1" data-cols="2" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">1 × 2</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">2 Cameras</div>
                </button>
                <button class="grid-option-btn" data-rows="1" data-cols="3" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">1 × 3</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">3 Cameras</div>
                </button>
                <button class="grid-option-btn" data-rows="2" data-cols="2" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">2 × 2</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">4 Cameras</div>
                </button>
                <button class="grid-option-btn" data-rows="2" data-cols="3" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">2 × 3</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">6 Cameras</div>
                </button>
                <button class="grid-option-btn" data-rows="3" data-cols="3" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">3 × 3</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">9 Cameras</div>
                </button>
                <button class="grid-option-btn" data-rows="4" data-cols="4" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">4 × 4</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">16 Cameras</div>
                </button>
                <button id="grid-custom-toggle" style="padding:0.75rem 0.5rem;border:2px solid #e2e8f0;border-radius:10px;background:white;cursor:pointer;transition:all 0.2s;text-align:center;">
                    <div style="font-weight:700;font-size:0.9rem;color:#497A21;">⚙️</div>
                    <div style="font-size:0.65rem;color:#94a3b8;">Custom</div>
                </button>
            </div>

            <!-- Custom input (hidden by default) -->
            <div id="grid-custom-section" style="display:none;background:#f8faf5;border-radius:10px;padding:1rem;margin-bottom:1rem;">
                <p style="font-size:0.8rem;color:#64748b;margin:0 0 0.75rem;">Enter rows × columns:</p>
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <input id="grid-custom-rows" type="number" min="1" max="5" value="2" style="width:60px;padding:0.5rem;border:1px solid #d1d5db;border-radius:8px;text-align:center;font-size:1rem;font-weight:600;">
                    <span style="font-weight:700;color:#64748b;">×</span>
                    <input id="grid-custom-cols" type="number" min="1" max="5" value="2" style="width:60px;padding:0.5rem;border:1px solid #d1d5db;border-radius:8px;text-align:center;font-size:1rem;font-weight:600;">
                    <span id="grid-custom-total" style="font-size:0.8rem;color:#497A21;font-weight:600;margin-left:0.5rem;">= 4 cameras</span>
                    <button id="grid-custom-go" style="margin-left:auto;padding:0.5rem 1rem;background:#497A21;color:white;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:0.85rem;">Go</button>
                </div>
            </div>

            <button id="grid-dialog-cancel" style="width:100%;padding:0.5rem;border:none;background:#f1f5f9;border-radius:8px;color:#64748b;cursor:pointer;font-size:0.85rem;">Cancel</button>
        </div>
    `;

    document.body.appendChild(overlay);

    // Cancel
    overlay.querySelector('#grid-dialog-cancel').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });

    // Custom toggle
    overlay.querySelector('#grid-custom-toggle').addEventListener('click', () => {
        const section = overlay.querySelector('#grid-custom-section');
        section.style.display = section.style.display === 'none' ? 'block' : 'none';
    });

    // Custom input — live total display
    const rowsInput = overlay.querySelector('#grid-custom-rows');
    const colsInput = overlay.querySelector('#grid-custom-cols');
    const totalLabel = overlay.querySelector('#grid-custom-total');
    const updateTotal = () => {
        const r = parseInt(rowsInput.value) || 1;
        const c = parseInt(colsInput.value) || 1;
        totalLabel.textContent = `= ${r * c} cameras`;
    };
    rowsInput.addEventListener('input', updateTotal);
    colsInput.addEventListener('input', updateTotal);

    // Custom Go button
    overlay.querySelector('#grid-custom-go').addEventListener('click', () => {
        const r = parseInt(rowsInput.value) || 1;
        const c = parseInt(colsInput.value) || 1;
        overlay.remove();
        if (r === 1 && c === 1) {
            const idleCam = findIdleCamera();
            if (!idleCam) { alert('No idle camera available. Click "Add Camera" first.'); return; }
            uploadToCamera(idleCam);
        } else {
            handleGridUpload(r, c);
        }
    });

    // Preset button hover + click
    overlay.querySelectorAll('.grid-option-btn').forEach(btn => {
        btn.addEventListener('mouseenter', () => { btn.style.borderColor = '#497A21'; btn.style.background = '#f0fdf4'; });
        btn.addEventListener('mouseleave', () => { btn.style.borderColor = '#e2e8f0'; btn.style.background = 'white'; });
        btn.addEventListener('click', () => {
            const rows = parseInt(btn.dataset.rows);
            const cols = parseInt(btn.dataset.cols);
            overlay.remove();

            if (rows === 1 && cols === 1) {
                // Single camera — use existing flow
                const idleCam = findIdleCamera();
                if (!idleCam) {
                    alert('No idle camera available. Click "Add Camera" first.');
                    return;
                }
                uploadToCamera(idleCam);
            } else {
                handleGridUpload(rows, cols);
            }
        });
    });
}

async function handleGridUpload(rows, cols) {
    const totalCams = rows * cols;
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/mp4,video/avi,video/mov,video/*';
    input.multiple = true; // Allow selecting multiple files

    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        if (!files.length) return;

        // Limit to the grid size
        const filesToUse = files.slice(0, totalCams);

        if (filesToUse.length < totalCams) {
            if (!confirm(`You selected ${filesToUse.length} video(s) but the ${rows}×${cols} grid needs ${totalCams}. Continue with ${filesToUse.length} camera(s)?`)) {
                return;
            }
        }

        // Hide placeholder
        const placeholder = document.getElementById('add-camera-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        // Create a camera cell for each video and upload individually
        for (let i = 0; i < filesToUse.length; i++) {
            const file = filesToUse[i];
            try {
                // Create camera cell on backend
                multiCctvCameraCounter++;
                const label = `Camera ${multiCctvCameraCounter}`;
                const addForm = new FormData();
                addForm.append('label', label);

                const addResp = await fetch(getApiUrl(ENDPOINTS.MULTI_CCTV_ADD), { method: 'POST', body: addForm });
                const addData = await addResp.json();
                const camId = addData.camera_id;

                // Register in frontend state
                multiCctvCameras[camId] = { label, count: 0, status: 'processing' };
                renderCameraCell(camId, label);
                updateCameraStatus(camId, 'processing', 'Uploading...');

                // Upload video to this camera
                const uploadForm = new FormData();
                uploadForm.append('file', file);
                if (userId) uploadForm.append('user_id', userId);

                fetch(getApiUrl(`${ENDPOINTS.MULTI_CCTV_UPLOAD}/${camId}`), {
                    method: 'POST',
                    body: uploadForm
                }).then(resp => resp.json()).then(data => {
                    updateCameraStatus(camId, 'processing', 'Processing...');
                    pollCameraCounts(camId);
                }).catch(err => {
                    console.error(`Upload failed for camera ${camId}:`, err);
                    updateCameraStatus(camId, 'error', 'Upload failed');
                });

            } catch (error) {
                console.error('Failed to create camera cell:', error);
            }
        }

        updateMultiCctvCamCount();
        updateGridColumns();
    };
    input.click();
}

// Find the first idle camera
function findIdleCamera() {
    for (const [camId, cam] of Object.entries(multiCctvCameras)) {
        if (cam.status === 'idle') return camId;
    }
    return null;
}

async function addCameraCell() {
    try {
        multiCctvCameraCounter++;
        const label = `Camera ${multiCctvCameraCounter}`;
        const formData = new FormData();
        formData.append('label', label);

        const response = await fetch(getApiUrl(ENDPOINTS.MULTI_CCTV_ADD), {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.camera_id) {
            multiCctvCameras[data.camera_id] = { label: data.label || label, count: 0, status: 'idle' };
            renderCameraCell(data.camera_id, data.label || label);
            updateMultiCctvCamCount();
            updateGridColumns();

            // Hide placeholder
            const placeholder = document.getElementById('add-camera-placeholder');
            if (placeholder) placeholder.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to add camera:', error);
    }
}

// Adjust grid columns based on camera count
function updateGridColumns() {
    const grid = document.getElementById('multi-cctv-grid');
    if (!grid) return;
    const count = Object.keys(multiCctvCameras).length;
    if (count <= 1) {
        grid.style.gridTemplateColumns = '1fr';
    } else if (count <= 4) {
        grid.style.gridTemplateColumns = 'repeat(2, 1fr)';
    } else {
        grid.style.gridTemplateColumns = 'repeat(3, 1fr)';
    }
}

function renderCameraCell(cameraId, label) {
    const grid = document.getElementById('multi-cctv-grid');
    if (!grid) return;

    const cell = document.createElement('div');
    cell.className = 'camera-cell';
    cell.id = `cam-${cameraId}`;
    cell.innerHTML = `
        <div class="camera-cell-body" id="cam-body-${cameraId}">
            <div class="cam-overlay-label">${label.toUpperCase()}</div>
            <div class="cam-overlay-count" id="cam-count-${cameraId}">0 sacks</div>
            <div class="cam-overlay-status">
                <span class="camera-cell-status idle" id="cam-status-${cameraId}">Idle</span>
                <button class="camera-cell-remove" title="Remove" onclick="removeCameraCell('${cameraId}')">&times;</button>
            </div>
            <p class="cam-waiting-text">Waiting for feed...</p>
        </div>
    `;

    grid.appendChild(cell);
}

function uploadToCamera(cameraId) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/mp4,video/avi,video/mov';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Update status
        updateCameraStatus(cameraId, 'processing', 'Processing...');
        multiCctvCameras[cameraId].status = 'processing';

        const formData = new FormData();
        formData.append('file', file);
        if (userId) formData.append('user_id', userId);

        try {
            const response = await fetch(`${getApiUrl(ENDPOINTS.MULTI_CCTV_UPLOAD)}/${cameraId}`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            console.log('Camera upload started:', data);

            // Poll for completion
            pollCameraCounts(cameraId);
        } catch (error) {
            console.error('Camera upload failed:', error);
            updateCameraStatus(cameraId, 'error', 'Error');
        }
    };
    input.click();
}

function startLiveCamera(cameraId) {
    // Auto-use webcam (source 0) — no prompt needed
    const source = '0';

    updateCameraStatus(cameraId, 'processing', 'Connecting...');
    multiCctvCameras[cameraId].status = 'live';

    const formData = new FormData();
    formData.append('source', source);

    fetch(`${getApiUrl(ENDPOINTS.MULTI_CCTV_LIVE)}/${cameraId}`, {
        method: 'POST',
        body: formData
    }).then(response => response.json()).then(data => {
        if (data.status === 'live_started') {
            updateCameraStatus(cameraId, 'live', 'LIVE');

            // Set MJPEG stream
            const cellBody = document.querySelector(`#cam-body-${cameraId}`);
            if (cellBody) {
                // Keep overlays, add img behind them
                const waitText = cellBody.querySelector('.cam-waiting-text');
                if (waitText) waitText.remove();
                const img = document.createElement('img');
                img.src = `${getApiUrl(ENDPOINTS.MULTI_CCTV_STREAM)}/${cameraId}?t=${Date.now()}`;
                img.style.cssText = 'width:100%;height:100%;object-fit:cover;position:absolute;top:0;left:0;';
                cellBody.insertBefore(img, cellBody.firstChild);
            }

            // Poll counts
            pollCameraCounts(cameraId);
        }
    }).catch(err => {
        console.error('Live start failed:', err);
        updateCameraStatus(cameraId, 'error', 'Error');
    });
}

// Expose functions to global scope for inline onclick handlers (module-scoped by default)
window.uploadToCamera = uploadToCamera;
window.startLiveCamera = startLiveCamera;
window.removeCameraCell = removeCameraCell;

async function removeCameraCell(cameraId) {
    try {
        await fetch(`${getApiUrl(ENDPOINTS.MULTI_CCTV_REMOVE)}/${cameraId}`, { method: 'POST' });
    } catch (e) {
        console.warn('Remove API failed:', e);
    }
    const cell = document.getElementById(`cam-${cameraId}`);
    if (cell) cell.remove();
    delete multiCctvCameras[cameraId];
    updateMultiCctvCamCount();
    updateMultiCctvTotal();
    updateGridColumns();

    // Show placeholder if no cameras left
    if (Object.keys(multiCctvCameras).length === 0) {
        const placeholder = document.getElementById('add-camera-placeholder');
        if (placeholder) placeholder.style.display = 'flex';
    }
}

function updateCameraStatus(cameraId, statusClass, text) {
    const el = document.querySelector(`#cam-status-${cameraId}`);
    if (el) {
        el.className = `camera-cell-status ${statusClass}`;
        el.textContent = text;
    }
}

function pollCameraCounts(cameraId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(getApiUrl(ENDPOINTS.MULTI_CCTV_COUNTS));
            const data = await response.json();

            if (data.cameras && data.cameras[cameraId]) {
                const cam = data.cameras[cameraId];
                const countEl = document.querySelector(`#cam-count-${cameraId}`);
                if (countEl) countEl.textContent = `${cam.count} sacks`;

                if (cam.status === 'completed') {
                    updateCameraStatus(cameraId, 'completed', 'Done');

                    // Show output video if available
                    if (cam.video_url) {
                        const cellBody = document.querySelector(`#cam-body-${cameraId}`);
                        if (cellBody) {
                            const waitText = cellBody.querySelector('.cam-waiting-text');
                            if (waitText) waitText.remove();
                            const existingMedia = cellBody.querySelector('img, video');
                            if (existingMedia) existingMedia.remove();
                            const vid = document.createElement('video');
                            vid.controls = true;
                            vid.src = `${getApiUrl('')}${cam.video_url}`;
                            vid.style.cssText = 'width:100%;height:100%;object-fit:cover;position:absolute;top:0;left:0;';
                            cellBody.insertBefore(vid, cellBody.firstChild);
                        }

                        // Add Download & Analytics buttons below the camera cell
                        const cell = document.getElementById(`cam-${cameraId}`);
                        if (cell && !cell.querySelector('.cam-result-actions')) {
                            const camLabel = multiCctvCameras[cameraId]?.label || `Camera ${cameraId}`;
                            const videoUrl = `${getApiUrl('')}${cam.video_url}`;
                            const actionsDiv = document.createElement('div');
                            actionsDiv.className = 'cam-result-actions';
                            actionsDiv.style.cssText = 'display:flex;gap:6px;padding:6px 8px;background:rgba(0,0,0,0.03);border-top:1px solid #e2e8f0;';
                            actionsDiv.innerHTML = `
                                <button class="btn-primary" style="flex:1;font-size:0.75rem;padding:5px 8px;" data-url="${videoUrl}" data-filename="detected_${camLabel}.mp4">📥 Download</button>
                                <button class="btn-primary" style="flex:1;font-size:0.75rem;padding:5px 8px;" data-label="${camLabel}">📊 Analytics</button>
                            `;
                            cell.appendChild(actionsDiv);

                            // Download handler
                            actionsDiv.querySelector('[data-url]').addEventListener('click', async (e) => {
                                const url = e.target.getAttribute('data-url');
                                const filename = e.target.getAttribute('data-filename');
                                try {
                                    const resp = await fetch(url);
                                    const blob = await resp.blob();
                                    const blobUrl = window.URL.createObjectURL(blob);
                                    const link = document.createElement('a');
                                    link.href = blobUrl;
                                    link.download = filename;
                                    document.body.appendChild(link);
                                    link.click();
                                    link.remove();
                                    window.URL.revokeObjectURL(blobUrl);
                                } catch (err) { console.error('Download failed:', err); }
                            });

                            // Analytics handler
                            actionsDiv.querySelector('[data-label]').addEventListener('click', (e) => {
                                localStorage.setItem('selectedAnalyticsFilter', e.target.getAttribute('data-label'));
                                window.location.href = 'analytics.html';
                            });
                        }
                    }
                    multiCctvCameras[cameraId].status = 'completed';

                    // Save to analytics so the Analytics page shows this data
                    const camLabel = multiCctvCameras[cameraId]?.label || `Camera ${cameraId}`;
                    addAnalyticsRow(camLabel, cam.count, "Completed");
                    saveRecentUpload({
                        fileName: camLabel,
                        count: cam.count,
                        mediaUrl: cam.video_url ? `${getApiUrl('')}${cam.video_url}` : '',
                        isImage: false,
                        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    });

                    clearInterval(interval);
                }

                multiCctvCameras[cameraId] = cam;
                updateMultiCctvTotal();
            }
        } catch (e) {
            console.error('Polling error:', e);
            clearInterval(interval);
        }
    }, 2000);
}

function updateMultiCctvTotal() {
    let total = 0;
    const statsList = document.getElementById('multi-cctv-stats-list');
    const grandTotalEl = document.getElementById('multi-cctv-grand-total');
    const statsContainer = document.getElementById('multi-cctv-stats');

    if (statsList) statsList.innerHTML = ''; // Clear previous chips

    const cameras = Object.values(multiCctvCameras);

    if (cameras.length > 0 && statsContainer) {
        statsContainer.style.display = 'block';
    } else if (statsContainer) {
        statsContainer.style.display = 'none';
    }

    cameras.forEach(cam => {
        const camCount = cam.count || 0;
        total += camCount;

        // Create an individual chip for each camera
        if (statsList) {
            const chip = document.createElement('div');
            chip.style.cssText = 'background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; font-size: 13px; color: #334155; display: flex; align-items: center; gap: 8px;';
            chip.innerHTML = `
                <strong>${cam.label}:</strong>
                <span style="background: #497A21; color: white; padding: 2px 8px; border-radius: 12px; font-weight: bold;">${camCount}</span>
            `;
            statsList.appendChild(chip);
        }
    });

    // Update the existing header total badge
    const el = document.getElementById('multi-cctv-total');
    if (el) el.textContent = total;

    // Update the new grand total at the bottom
    if (grandTotalEl) grandTotalEl.textContent = total;
}

function updateMultiCctvCamCount() {
    const el = document.getElementById('multi-cctv-cam-count');
    if (el) el.textContent = `${Object.keys(multiCctvCameras).length} Camera${Object.keys(multiCctvCameras).length !== 1 ? 's' : ''}`;
}

// ============================================================
// GODOWN MODE LOGIC
// ============================================================

async function loadGodownStatus() {
    try {
        const response = await fetch(getApiUrl(ENDPOINTS.GODOWN_STATUS));
        const data = await response.json();
        updateGodownStats(data);
    } catch (e) {
        console.error('Failed to load godown status:', e);
    }
}

function updateGodownStats(data) {
    const inv = document.getElementById('godown-inventory');
    const todayIn = document.getElementById('godown-today-in');
    const todayOut = document.getElementById('godown-today-out');
    const netTrend = document.getElementById('godown-net-trend');

    if (inv && data.inventory !== undefined) inv.textContent = data.inventory;
    if (todayIn && data.today_in !== undefined) todayIn.textContent = data.today_in;
    if (todayOut && data.today_out !== undefined) todayOut.textContent = data.today_out;

    if (netTrend) {
        const net = (data.today_in || 0) - (data.today_out || 0);
        netTrend.textContent = `Net: ${net >= 0 ? '+' : ''}${net}`;
        netTrend.style.color = net >= 0 ? '#00C853' : '#FF5252';
    }
}
