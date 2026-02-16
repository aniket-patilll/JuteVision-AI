import { requireAuth, signOut } from './auth.js';

// Protect Route
requireAuth();

// Logout Logic
const logoutBtn = document.getElementById('nav-logout');
if (logoutBtn) {
    logoutBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        await signOut();
        window.location.href = 'login.html';
    });
}

function updateDate() {
    const dateElement = document.getElementById('current-date');
    if (dateElement) {
        const now = new Date();
        const options = { month: 'short', day: 'numeric', year: 'numeric' };
        const dateString = now.toLocaleDateString('en-US', options);
        dateElement.textContent = `Today: ${dateString}`;
    }
}

// Global State
const state = {
    allData: [],
    filteredData: []
};

// Colors
const colors = {
    brandGreen: '#497A21',
    brandGreenLight: 'rgba(73, 122, 33, 0.7)',
    brandGreenTransparent: 'rgba(73, 122, 33, 0.2)',
    textSecondary: '#64748b',
    danger: '#ef4444'
};

// Initialize Dashboard
function initDashboard() {
    loadAnalyticsData();
    updateDate();
    // Update date every minute to keep it "real-time"
    setInterval(updateDate, 60000);

    // Listen for filter changes
    const filterSelect = document.getElementById('file-filter');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            handleFilterChange(e.target.value);
        });
    }

    // Listen for storage events (updates from other tabs/pages)
    window.addEventListener('storage', (e) => {
        if (e.key === 'analyticsData') {
            loadAnalyticsData();
        }
    });

    // Export Button
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }
}

// Load Data
function loadAnalyticsData() {
    const storedData = localStorage.getItem('analyticsData');
    if (storedData) {
        state.allData = JSON.parse(storedData) || [];
    } else {
        state.allData = [];
    }

    // Initial Filter (Check for redirect filter first)
    const redirectFilter = localStorage.getItem('selectedAnalyticsFilter');

    populateFilterDropdown();

    if (redirectFilter) {
        // Set dropdown value if it exists in options
        const filterSelect = document.getElementById('file-filter');
        if (filterSelect && [...filterSelect.options].some(opt => opt.value === redirectFilter)) {
            filterSelect.value = redirectFilter;
            handleFilterChange(redirectFilter);
        } else {
            // Fallback if file not found (unlikely but safe)
            handleFilterChange('all');
        }
        // Clear it so navigation doesn't get stuck on this filter
        localStorage.removeItem('selectedAnalyticsFilter');
    } else {
        handleFilterChange('all');
    }
}

// Populate Dropdown
function populateFilterDropdown() {
    const filterSelect = document.getElementById('file-filter');
    if (!filterSelect) return;

    // Save current selection if re-populating
    const currentSelection = filterSelect.value;

    // Get unique filenames
    const fileNames = [...new Set(state.allData.map(item => item.filename))];

    // Clear existing options except "All"
    filterSelect.innerHTML = '<option value="all">All Processing</option>';

    fileNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        filterSelect.appendChild(option);
    });

    // Restore selection if it still exists
    if (fileNames.includes(currentSelection)) {
        filterSelect.value = currentSelection;
    }
}

// Handle Filter Selection
function handleFilterChange(filterValue) {
    if (filterValue === 'all') {
        state.filteredData = [...state.allData];
    } else {
        state.filteredData = state.allData.filter(item => item.filename === filterValue);
    }

    // Render everything with filtered data
    renderDashboard(state.filteredData);
}


// Master Render Function
function renderDashboard(data) {
    renderTable(data);
    updateGlobalStats(data);

    // Update Charts
    updateProductionChart(data);
    updateStatusChart(data);

    // Initialize all charts
    initWeeklyChart(data);
    initRadarChart(data);
    initHeatmap(data);
    initSizeChart(data);
    initSpeedChart(data);
    initSourceChart(data);
}

// Render Table
function renderTable(data) {
    const tableBody = document.getElementById('analytics-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    if (data.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No data matching selection</td>
            </tr>
        `;
        return;
    }

    data.forEach(item => {
        const row = document.createElement('tr');

        // Determine Status Style
        let badgeClass = 'status-info';
        let statusText = item.status;
        if (item.status === 'Completed' || item.status === 'Verified') {
            badgeClass = 'status-success';
        } else if (item.status === 'Failed') {
            badgeClass = 'status-failed';
        }

        // Determine File Icon (Video vs Image)
        const isVideo = item.filename && (item.filename.endsWith('.mp4') || item.filename.endsWith('.avi'));
        const fileIcon = isVideo
            ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg>`
            : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>`;

        row.innerHTML = `
            <td style="color: var(--text-secondary); font-variant-numeric: tabular-nums;">${item.time}</td>
            <td>
                <div class="file-cell">
                    <div class="file-icon">${fileIcon}</div>
                    <span>${item.filename}</span>
                </div>
            </td>
            <td style="font-weight: 600;">${item.count} Bags</td>
            <td><span class="status-badge ${badgeClass}">${statusText}</span></td>
            <td>
                <div class="action-cell">
                    <button class="action-btn" title="View Report">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                    </button>
                    <button class="action-btn" title="Download CSV">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    </button>
                    <button class="action-btn delete-btn" title="Delete" style="color: var(--danger);">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                    </button>
                </div>
            </td>
        `;
        tableBody.appendChild(row);
    });

    // Re-attach event listeners for dynamic rows
    attachTableListeners();
}

function attachTableListeners() {
    const tableBodyElement = document.getElementById('analytics-table-body');
    if (!tableBodyElement) return;

    // Remove old listener if any (simplest way is to clone and replace, but we just re-rendered innerHTML so it's fresh)
    // Actually, event delegation on parent is better, moved to initDashboard in global scope? 
    // No, initDashboard runs once. Let's assume we use the one attached to body or manage here.
    // For safety, let's keep the delegation logic simple.

    // NOTE: The previous code attached listener to tableBodyElement.
    // Since we cleared innerHTML, the listener on the element itself (if added via addEventListener) PERSISTS.
    // So we don't need to re-attach if we attach to the parent (tableBody) once.
}

// Attach listeners ONCE
const tableBody = document.getElementById('analytics-table-body');
if (tableBody) {
    tableBody.addEventListener('click', (e) => {
        const btn = e.target.closest('.action-btn');
        if (!btn) return;

        const row = btn.closest('tr');
        const fileName = row.querySelector('.file-cell span').textContent;
        const count = row.cells[2].textContent;
        const status = row.cells[3].textContent.trim();
        const time = row.cells[0].textContent;

        if (btn.title === 'View Report') {
            alert(`Report Details:\n\nFile: ${fileName}\nTime: ${time}\nBags: ${count}\nStatus: ${status}`);
        } else if (btn.title === 'Download CSV') {
            const csvContent = `data:text/csv;charset=utf-8,Time,File,Count,Status\n${time},${fileName},${count},${status}`;
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `${fileName}_report.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else if (btn.title === 'Delete') {
            if (confirm(`Are you sure you want to delete the record for "${fileName}"?`)) {
                // Remove from state
                const index = state.allData.findIndex(item => item.time === time && item.filename === fileName);
                if (index > -1) {
                    state.allData.splice(index, 1);

                    // Update LocalStorage
                    localStorage.setItem('analyticsData', JSON.stringify(state.allData));

                    // Re-render
                    // If a filter is active, re-apply it
                    const filterSelect = document.getElementById('file-filter');
                    handleFilterChange(filterSelect ? filterSelect.value : 'all');

                    // Also update the filter dropdown in case we deleted the last entry of a file
                    populateFilterDropdown();
                }
            }
        }
    });
}


// Update KPI Cards
function updateGlobalStats(data) {
    const totalUploads = data.length;
    let totalBags = 0;
    let successCount = 0;

    data.forEach(item => {
        const count = parseInt(item.count) || 0;
        const status = item.status || '';

        totalBags += count;
        if (status.includes('Completed') || status.includes('Verified')) {
            successCount++;
        }
    });

    const avgBags = totalUploads > 0 ? Math.round(totalBags / totalUploads) : 0;
    const successRate = totalUploads > 0 ? Math.round((successCount / totalUploads) * 100) : 100;

    // v9.5 Calculate Peak Hour (Hour with most bags)
    const hourlyCounts = new Array(24).fill(0);
    data.forEach(item => {
        if (item.time && item.count) {
            const hour = parseInt(item.time.split(':')[0]);
            if (!isNaN(hour)) hourlyCounts[hour] += (parseInt(item.count) || 0);
        }
    });

    let peakHour = 0;
    let maxBags = -1;
    hourlyCounts.forEach((count, hour) => {
        if (count > maxBags) {
            maxBags = count;
            peakHour = hour;
        }
    });

    const peakTimeString = maxBags > 0 ? `${peakHour.toString().padStart(2, '0')}:00` : '--:--';

    // Update DOM
    if (document.getElementById('metric-uploads'))
        document.getElementById('metric-uploads').textContent = totalUploads;

    if (document.getElementById('metric-total-bags'))
        document.getElementById('metric-total-bags').textContent = totalBags;

    if (document.getElementById('metric-avg'))
        document.getElementById('metric-avg').textContent = avgBags;

    if (document.getElementById('metric-success'))
        document.getElementById('metric-success').textContent = `${successRate}%`;

    if (document.getElementById('metric-peak'))
        document.getElementById('metric-peak').textContent = peakTimeString;

    // Visibility Logic
    const filterSelect = document.getElementById('file-filter');
    const filterValue = filterSelect ? filterSelect.value : 'all';
    const isShowingAll = filterValue === 'all';

    const cardTotalBags = document.getElementById('card-total-bags');
    const cardAvg = document.getElementById('card-avg');
    const cardSuccess = document.getElementById('card-success');

    if (cardTotalBags) cardTotalBags.style.display = 'flex'; // Always visible as per user request
    if (cardAvg) cardAvg.style.display = isShowingAll ? 'flex' : 'none';
    if (cardSuccess) cardSuccess.style.display = isShowingAll ? 'flex' : 'none';
}


// --- Charts ---

let productionChartInstance = null;
let statusChartInstance = null;

// 1. Production Trend Chart
function updateProductionChart(data) {
    const ctx = document.getElementById('productionChart');
    if (!ctx) return;

    // Aggregate data by hour
    const hourlyCounts = new Array(24).fill(0);
    data.forEach(item => {
        if (item.time && item.count) {
            const [hours] = item.time.split(':').map(Number);
            if (!isNaN(hours) && hours >= 0 && hours < 24) {
                hourlyCounts[hours] += parseInt(item.count);
            }
        }
    });

    // Generate Labels (Last 12 hours relative to now)
    const labels = [];
    const values = [];
    const currentHour = new Date().getHours();
    for (let i = 11; i >= 0; i--) {
        let h = currentHour - i;
        if (h < 0) h += 24;
        labels.push(`${h}:00`);
        values.push(hourlyCounts[h]);
    }

    if (productionChartInstance) productionChartInstance.destroy();

    // Create Gradient
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(73, 122, 33, 0.5)'); // Brand Green
    gradient.addColorStop(1, 'rgba(73, 122, 33, 0.0)');

    productionChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Processed Bags',
                data: values,
                borderColor: colors.brandGreen,
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: '#fff',
                pointBorderColor: colors.brandGreen,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#1e293b',
                    bodyColor: '#1e293b',
                    borderColor: '#e2e8f0',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#f1f5f9', borderDash: [5, 5] },
                    ticks: { color: colors.textSecondary }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: colors.textSecondary }
                }
            }
        }
    });
}

// 2. Status Distribution Chart
function updateStatusChart(data) {
    const ctx = document.getElementById('statusChart');
    if (!ctx) return;

    let success = 0, failed = 0;
    data.forEach(item => {
        const s = (item.status || '').toLowerCase();
        if (s.includes('completed') || s.includes('verified')) success++;
        else failed++;
    });

    // If no data, show empty ring or 0
    if (data.length === 0) {
        success = 0; failed = 0;
    }

    if (statusChartInstance) statusChartInstance.destroy();

    // Update Center Text
    const rate = (success + failed) > 0 ? Math.round((success / (success + failed)) * 100) : 100;
    const centerText = document.getElementById('donut-total');
    if (centerText) centerText.textContent = `${rate}%`;

    statusChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Success', 'Failed'],
            datasets: [{
                data: [success, failed],
                backgroundColor: [colors.brandGreen, colors.danger],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, padding: 20, color: colors.textSecondary }
                }
            }
        }
    });
}

// --- Mock/Static Charts (Now Dynamic) ---

function initWeeklyChart(data) {
    const ctx = document.getElementById('weeklyChart');
    if (!ctx) return;

    // Calculate total bags for "Today" (Assuming Friday/Saturday for demo)
    let totalBags = 0;
    if (data && data.length > 0) {
        data.forEach(item => totalBags += (parseInt(item.count) || 0));
    }

    // v9.6 Dynamic Day Alignment (Monday = 0, Sunday = 6)
    const todayIndex = (new Date().getDay() + 6) % 7;

    // Historical Mock Data (Mon-Sun)
    const historical = [120, 150, 180, 200, 160, 210, 0];

    // Replace today's slot with real data
    historical[todayIndex] = totalBags;

    // Clear future slots
    for (let i = todayIndex + 1; i < 7; i++) historical[i] = 0;

    // Destroy existing
    const existing = Chart.getChart(ctx);
    if (existing) existing.destroy();

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Bags',
                data: historical,
                backgroundColor: (context) => {
                    const index = context.dataIndex;
                    return index === todayIndex ? colors.brandGreen : colors.brandGreenLight;
                },
                borderRadius: 4,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { display: false } },
                x: { grid: { display: false } }
            }
        }
    });
}

function initRadarChart(data) {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;

    // Update metrics based on data presence
    // If data is empty, metrics drop to 0
    let accuracy = 0, speed = 0, uptime = 0, throughput = 0, reliability = 0;

    if (data && data.length > 0) {
        // Simple mock logic: base + random variation
        accuracy = 95;
        speed = 88;
        uptime = 99;
        throughput = 90;
        reliability = 96;
    }

    const existing = Chart.getChart(ctx);
    if (existing) existing.destroy();

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Speed', 'Uptime', 'Throughput', 'Reliability'],
            datasets: [{
                label: 'System Health',
                data: [accuracy, speed, uptime, throughput, reliability],
                backgroundColor: colors.brandGreenTransparent,
                borderColor: colors.brandGreen,
                pointBackgroundColor: colors.brandGreen,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(0,0,0,0.1)' },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                    pointLabels: { font: { size: 10 } },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function initHeatmap(data) {
    const container = document.getElementById('heatmapGrid');
    if (!container) return;

    container.innerHTML = '';

    // If no data, render empty grid
    const hasData = data && data.length > 0;

    // Build Hourly Map from Data
    const hourlyActivity = new Array(24).fill(0);
    if (hasData) {
        data.forEach(item => {
            if (item.time) {
                const [h] = item.time.split(':').map(Number);
                if (!isNaN(h)) hourlyActivity[h]++;
            }
        });
    }

    // Header Row (Hours)
    container.appendChild(document.createElement('div'));
    for (let h = 0; h < 24; h++) {
        const header = document.createElement('div');
        header.className = 'heatmap-header';
        header.textContent = h % 6 === 0 ? h : '';
        container.appendChild(header);
    }

    // Rows (Mon-Sun) - Only "Today" (Sat) gets real data? or simulate?
    // Let's simulate: Sat = Real Data, others = Mock low activity or random
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    days.forEach((day, index) => {
        const label = document.createElement('div');
        label.className = 'heatmap-label';
        label.textContent = day;
        container.appendChild(label);

        for (let h = 0; h < 24; h++) {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';

            let intensity = 0;
            // v9.6 Use current system day for real data
            const todayIndex = (new Date().getDay() + 6) % 7;

            if (index === todayIndex && hasData) {
                // Normalize: max activity = 1.0 (e.g. 5 uploads/hour)
                intensity = Math.min(hourlyActivity[h] / 5, 1.0);
            } else if (hasData && index < todayIndex) {
                // Background noise for other days if data exists
                if (h >= 9 && h <= 17) intensity = Math.random() * 0.3;
            }

            // If no data at all, intensity stays 0

            cell.style.backgroundColor = `rgba(73, 122, 33, ${intensity})`;
            cell.title = `${day} ${h}:00 - Activity: ${hasData ? Math.round(intensity * 100) : 0}%`;
            container.appendChild(cell);
        }
    });
}

function initSizeChart(data) {
    const ctx = document.getElementById('sizeChart');
    if (!ctx) return;

    // Distribute Total Count
    let totalBags = 0;
    if (data && data.length > 0) {
        data.forEach(item => totalBags += (parseInt(item.count) || 0));
    }

    const small = Math.round(totalBags * 0.3);
    const medium = Math.round(totalBags * 0.5);
    const large = Math.round(totalBags * 0.2);

    const existing = Chart.getChart(ctx);
    if (existing) existing.destroy();

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Small', 'Medium', 'Large'],
            datasets: [{
                data: [small, medium, large],
                backgroundColor: ['#86efac', '#4ade80', '#16a34a'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { usePointStyle: true, boxWidth: 8 } }
            }
        }
    });
}

function initSpeedChart(data) {
    const ctx = document.getElementById('speedChart');
    if (!ctx) return;

    // If no data, empty chart
    let speeds = [];
    if (data && data.length > 0) {
        // Mock improving speed based on count
        speeds = [120, 115, 110, 108, 105, 102];
    } else {
        speeds = [0, 0, 0, 0, 0, 0];
    }

    const existing = Chart.getChart(ctx);
    if (existing) existing.destroy();

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6'],
            datasets: [{
                label: 'Time (ms)',
                data: speeds,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { borderDash: [5, 5] }, beginAtZero: true },
                x: { grid: { display: false } }
            }
        }
    });
}

function initSourceChart(data) {
    const ctx = document.getElementById('sourceChart');
    if (!ctx) return;

    // Efficiency only if data exists
    let webcam = 0, uploads = 0;
    if (data && data.length > 0) {
        // Mock efficiency data for now
        webcam = 92;
        uploads = 95;
    }

    const existing = Chart.getChart(ctx);
    if (existing) existing.destroy();

    new Chart(ctx, {
        type: 'bar',
        indexAxis: 'y',
        data: {
            labels: ['Webcam', 'Uploads'],
            datasets: [{
                label: 'Efficiency Score',
                data: [webcam, uploads],
                backgroundColor: '#f97316',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { max: 100, grid: { display: false } },
                y: { grid: { display: false } }
            }
        }
    });
}

function exportData() {
    const dataToExport = state.filteredData;
    if (dataToExport.length === 0) {
        alert("No data to export");
        return;
    }

    const csvContent = "data:text/csv;charset=utf-8,Time,File,Count,Status\n" +
        dataToExport.map(row =>
            `${row.time},${row.filename},${row.count},${row.status}`
        ).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "cctv_visioncount_analytics.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Start
initDashboard();
