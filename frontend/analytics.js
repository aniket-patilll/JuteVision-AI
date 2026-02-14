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
                const badgeClass = item.status === 'Completed' ? 'success' : 'info';
                row.innerHTML = `
                    <td>${item.time}</td>
                    <td>${item.filename}</td>
                    <td>${item.count}</td>
                    <td><span class="status-badge ${badgeClass}">${item.status}</span></td>
                `;
                tableBody.appendChild(row);
            });
        }
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

// Export Data
const exportBtn = document.getElementById('export-btn');
if (exportBtn) {
    exportBtn.addEventListener('click', () => {
        const tableBody = document.getElementById('analytics-table-body');
        const rows = Array.from(tableBody.querySelectorAll('tr'));

        // CSV Header
        let csvContent = "data:text/csv;charset=utf-8,Time,File,Count,Status\\n";

        rows.forEach(row => {
            // Skip empty state
            if (row.cells.length <= 1) return;

            const cols = Array.from(row.querySelectorAll('td'))
                .map(td => td.textContent.trim())
                .join(",");
            csvContent += cols + "\\r\\n";
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

// Load data on page load
loadAnalyticsData();

// Listen for storage events (updates from other tabs/pages)
window.addEventListener('storage', (e) => {
    if (e.key === 'analyticsData') {
        loadAnalyticsData();
    }
});
