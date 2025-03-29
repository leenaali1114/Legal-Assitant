document.addEventListener('DOMContentLoaded', function() {
    // Load analytics data
    loadAnalyticsData();
});

function loadAnalyticsData() {
    fetch('/get_analytics_data')
        .then(response => response.json())
        .then(data => {
            // Update stats
            document.getElementById('total-cases').textContent = Object.values(data.case_types).reduce((a, b) => a + b, 0);
            document.getElementById('case-types').textContent = Object.keys(data.case_types).length;
            document.getElementById('courts').textContent = Object.keys(data.courts).length;
            document.getElementById('judges').textContent = '5+'; // Placeholder
            
            // Create charts
            createCaseTypesChart(data.case_types);
            createCourtsChart(data.courts);
            createVerdictsChart(data.verdicts);
            createAppealsChart(data.appeals);
            createTimeChart(data.years);
            
            // Generate insights
            generateInsights(data);
        })
        .catch(error => {
            console.error('Error loading analytics data:', error);
        });
}

function createCaseTypesChart(data) {
    const ctx = document.getElementById('case-types-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Number of Cases',
                data: Object.values(data),
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createCourtsChart(data) {
    const ctx = document.getElementById('courts-chart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: [
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(155, 89, 182, 0.7)',
                    'rgba(241, 196, 15, 0.7)',
                    'rgba(231, 76, 60, 0.7)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(155, 89, 182, 1)',
                    'rgba(241, 196, 15, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true
        }
    });
}

function createVerdictsChart(data) {
    const ctx = document.getElementById('verdicts-chart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: [
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(231, 76, 60, 0.7)',
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(155, 89, 182, 0.7)',
                    'rgba(241, 196, 15, 0.7)'
                ],
                borderColor: [
                    'rgba(46, 204, 113, 1)',
                    'rgba(231, 76, 60, 1)',
                    'rgba(52, 152, 219, 1)',
                    'rgba(155, 89, 182, 1)',
                    'rgba(241, 196, 15, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true
        }
    });
}

function createAppealsChart(data) {
    const ctx = document.getElementById('appeals-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Number of Cases',
                data: Object.values(data),
                backgroundColor: 'rgba(155, 89, 182, 0.7)',
                borderColor: 'rgba(155, 89, 182, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createTimeChart(data) {
    const ctx = document.getElementById('time-chart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Number of Cases',
                data: Object.values(data),
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function generateInsights(data) {
    const insightsContainer = document.getElementById('insights-content');
    
    // Clear existing insights
    insightsContainer.innerHTML = '';
    
    // Calculate insights
    const totalCases = Object.values(data.case_types).reduce((a, b) => a + b, 0);
    const topCaseType = Object.entries(data.case_types).sort((a, b) => b[1] - a[1])[0];
    const topCourt = Object.entries(data.courts).sort((a, b) => b[1] - a[1])[0];
    const topVerdict = Object.entries(data.verdicts).sort((a, b) => b[1] - a[1])[0];
    const appealRate = (data.appeals['Appeal Pending'] + data.appeals['Appeal Dismissed']) / totalCases * 100;
    
    // Create insight items
    const insights = [
        {
            title: 'Case Type Distribution',
            content: `The most common case type is "${topCaseType[0]}" with ${topCaseType[1]} cases (${(topCaseType[1] / totalCases * 100).toFixed(1)}% of total).`
        },
        {
            title: 'Court Activity',
            content: `"${topCourt[0]}" handles the most cases with ${topCourt[1]} cases (${(topCourt[1] / totalCases * 100).toFixed(1)}% of total).`
        },
        {
            title: 'Verdict Patterns',
            content: `The most common verdict is "${topVerdict[0]}" with ${topVerdict[1]} cases (${(topVerdict[1] / totalCases * 100).toFixed(1)}% of total).`
        },
        {
            title: 'Appeal Rate',
            content: `Approximately ${appealRate.toFixed(1)}% of cases are appealed.`
        },
        {
            title: 'Case Trends',
            content: `The number of cases has ${Object.values(data.years)[Object.values(data.years).length - 1] > Object.values(data.years)[0] ? 'increased' : 'decreased'} over time.`
        }
    ];
    
    // Add insights to container
    insights.forEach(insight => {
        const insightDiv = document.createElement('div');
        insightDiv.className = 'insight-item';
        
        const insightTitle = document.createElement('h5');
        insightTitle.textContent = insight.title;
        
        const insightContent = document.createElement('p');
        insightContent.textContent = insight.content;
        
        insightDiv.appendChild(insightTitle);
        insightDiv.appendChild(insightContent);
        insightsContainer.appendChild(insightDiv);
    });
} 