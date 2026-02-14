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

// Load analytics data from localStorage or backend
function loadAnalyticsData() {
    const tableBody = document.getElementById('analytics-table-body');

    // Try to get data from localStorage (shared from main dashboard)
    const storedData = localStorage.getItem('analyticsData');
    if (storedData) {
        const data = JSON.parse(storedData);
        if (data && data.length > 0) {
            tableBody.innerHTML = '';
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
                const isVideo = item.filename.endsWith('.mp4') || item.filename.endsWith('.avi');
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
                        </div>
                    </td>
                `;
                tableBody.appendChild(row);
            });

            // Update Charts
            updateProductionChart(data);
            updateStatusChart(data);
        }
    }

    // Event Delegation for Table Actions (View/Download)
    const tableBodyElement = document.getElementById('analytics-table-body');
    if (tableBodyElement) {
        tableBodyElement.addEventListener('click', (e) => {
            const btn = e.target.closest('.action-btn');
            if (!btn) return;

            const row = btn.closest('tr');
            const fileName = row.querySelector('.file-cell span').textContent;
            const count = row.cells[2].textContent;
            const status = row.cells[3].textContent.trim();
            const time = row.cells[0].textContent;

            if (btn.title === 'View Report') {
                alert(`Report Details:\n\nFile: ${fileName}\nTime: ${time}\nBags: ${count}\nStatus: ${status}\n\n(Full report view coming soon)`);
            } else if (btn.title === 'Download CSV') {
                const csvContent = `data:text/csv;charset=utf-8,Time,File,Count,Status\n${time},${fileName},${count},${status}`;
                const encodedUri = encodeURI(csvContent);
                const link = document.createElement("a");
                link.setAttribute("href", encodedUri);
                link.setAttribute("download", `${fileName}_report.csv`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        });
    }
    updateGlobalStats();
}

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

// Chart Instances
let productionChartInstance = null;
let statusChartInstance = null;

// Update Production Chart Logic (Gradient Line Chart)
function updateProductionChart(data) {
    const ctx = document.getElementById('productionChart');
    if (!ctx) return;

    const hourlyCounts = new Array(24).fill(0);
    data.forEach(item => {
        if (item.time && item.count) {
            const [hours] = item.time.split(':').map(Number);
            if (!isNaN(hours)) hourlyCounts[hours] += parseInt(item.count);
        }
    });

    // Last 12 Hours
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
        type: 'line', // Switch to Line
        data: {
            labels: labels,
            datasets: [{
                label: 'Processed Bags',
                data: values,
                borderColor: '#497A21',
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#497A21',
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4 // Smooth curves
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
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Update Status Chart Logic (Doughnut)
function updateStatusChart(data) {
    const ctx = document.getElementById('statusChart');
    if (!ctx) return;

    let success = 0, failed = 0;
    data.forEach(item => {
        const s = item.status.toLowerCase();
        if (s.includes('completed') || s.includes('verified')) success++;
        else failed++;
    });

    if (statusChartInstance) statusChartInstance.destroy();

    // Update Center Text
    const successRate = (success + failed) > 0 ? Math.round((success / (success + failed)) * 100) : 100;
    const centerText = document.getElementById('donut-total');
    if (centerText) centerText.textContent = `${successRate}%`;

    statusChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Success', 'Failed'],
            datasets: [{
                data: [success, failed],
                backgroundColor: ['#497A21', '#ef4444'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%', // Thinner ring
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, padding: 20, color: '#64748b' }
                }
            }
        }
    });
}

// Export Data
const exportBtn = document.getElementById('export-btn');
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

// --- NEW CHARTS IMPLEMENTATION ---

// 1. Weekly Performance Chart (FIXED: Green Theme)
function initWeeklyChart() {
    const ctx = document.getElementById('weeklyChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Bags',
                data: [120, 150, 180, 200, 160, 90, 100], // Mock Data
                backgroundColor: 'rgba(73, 122, 33, 0.7)', // Brand Green
                hoverBackgroundColor: '#497A21',
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

// 2. System Health Radar Chart
function initRadarChart() {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Speed', 'Uptime', 'Throughput', 'Reliability'],
            datasets: [{
                label: 'System Health',
                data: [95, 88, 99, 90, 96],
                backgroundColor: 'rgba(73, 122, 33, 0.2)',
                borderColor: '#497A21',
                pointBackgroundColor: '#497A21',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#497A21'
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
                    suggestedMin: 50,
                    suggestedMax: 100
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// 3. Activity Heatmap Generator
function initHeatmap() {
    const container = document.getElementById('heatmapGrid');
    if (!container) return;

    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    container.innerHTML = '';

    // Header Row (Hours)
    container.appendChild(document.createElement('div')); // Empty corner
    for (let h = 0; h < 24; h++) {
        const header = document.createElement('div');
        header.className = 'heatmap-header';
        header.textContent = h % 6 === 0 ? h : ''; // Show label every 6 hours
        container.appendChild(header);
    }

    // Rows
    days.forEach(day => {
        // Day Label
        const label = document.createElement('div');
        label.className = 'heatmap-label';
        label.textContent = day;
        container.appendChild(label);

        // Hour Cells
        for (let h = 0; h < 24; h++) {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';

            // Mock Intensity (Higher activity midday)
            let intensity = 0;
            if (h >= 9 && h <= 17) intensity = Math.random() * 0.5 + 0.5; // High
            else intensity = Math.random() * 0.3; // Low

            // Color: Green with variable opacity
            cell.style.backgroundColor = `rgba(73, 122, 33, ${intensity})`;
            cell.title = `${day} ${h}:00 - Activity: ${Math.round(intensity * 100)}%`;

            container.appendChild(cell);
        }
    });
}

// --- ROUND 2 NEW CHARTS ---

// 4. Bag Size Distribution (Pie)
function initSizeChart() {
    const ctx = document.getElementById('sizeChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Small', 'Medium', 'Large'],
            datasets: [{
                data: [30, 50, 20],
                backgroundColor: ['#86efac', '#4ade80', '#16a34a'], // Light to Dark Green
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

// 5. Processing Speed Trend (Line)
function initSpeedChart() {
    const ctx = document.getElementById('speedChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6'],
            datasets: [{
                label: 'Time (ms)',
                data: [120, 115, 110, 108, 105, 102], // Improving speed
                borderColor: '#3b82f6', // Blue for contrast
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
                y: { grid: { borderDash: [5, 5] } },
                x: { grid: { display: false } }
            }
        }
    });
}

// 6. Source Efficiency (Horizontal Bar)
function initSourceChart() {
    const ctx = document.getElementById('sourceChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        indexAxis: 'y', // Horizontal
        data: {
            labels: ['Cam 1', 'Cam 2', 'Uploads', 'Mobile'],
            datasets: [{
                label: 'Efficiency Score',
                data: [92, 85, 98, 75],
                backgroundColor: '#f97316', // Orange
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

// Initialize all charts
loadAnalyticsData();
initWeeklyChart();
initRadarChart();
initHeatmap();
initSizeChart();
initSpeedChart();
initSourceChart();

// Listen for storage events (updates from other tabs/pages)
window.addEventListener('storage', (e) => {
    if (e.key === 'analyticsData') {
        loadAnalyticsData();
    }
});
