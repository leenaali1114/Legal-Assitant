document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    const tabs = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs and sections
            tabs.forEach(t => t.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Show corresponding section
            const targetId = this.id.replace('-tab', '-section');
            document.getElementById(targetId).classList.add('active');
            
            // Load analytics data if analytics tab is clicked
            if (this.id === 'analytics-tab') {
                loadAnalyticsData();
            }
        });
    });
    
    // Chat functionality
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        userInput.value = '';
        
        // Show typing indicator
        addTypingIndicator();
        
        // Send message to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add assistant response to chat
            addMessageToChat('assistant', data.response);
            
            // Handle special response types
            if (data.type === 'similar_cases' && data.cases) {
                addCasesToChat(data.cases);
            } else if (data.type === 'case_type_prediction' && data.details) {
                addCaseTypeDetailsToChat(data.details);
            } else if (data.type === 'legal_reference' && data.details) {
                addLegalReferenceDetailsToChat(data.details);
            } else if (data.type === 'judge_info' && data.details) {
                addJudgeInfoToChat(data.details);
            } else if (data.type === 'court_info' && data.details) {
                addCourtInfoToChat(data.details);
            }
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessageToChat('assistant', 'Sorry, there was an error processing your request.');
            scrollToBottom();
        });
    }
    
    function addMessageToChat(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const paragraph = document.createElement('p');
        paragraph.textContent = message;
        
        contentDiv.appendChild(paragraph);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        scrollToBottom();
    }
    
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator-container';
        typingDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicatorDiv.appendChild(dot);
        }
        
        contentDiv.appendChild(indicatorDiv);
        typingDiv.appendChild(contentDiv);
        chatMessages.appendChild(typingDiv);
        
        scrollToBottom();
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addCasesToChat(cases) {
        const casesDiv = document.createElement('div');
        casesDiv.className = 'message assistant';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content cases-list';
        
        cases.forEach(caseItem => {
            const caseDiv = document.createElement('div');
            caseDiv.className = 'case-item';
            
            const caseTitle = document.createElement('h5');
            caseTitle.textContent = `Case ${caseItem.Case_ID}: ${caseItem.Case_Type}`;
            
            const caseCourt = document.createElement('p');
            caseCourt.innerHTML = `<strong>Court:</strong> ${caseItem.Court}`;
            
            const caseJudge = document.createElement('p');
            caseJudge.innerHTML = `<strong>Judge:</strong> ${caseItem.Judge_Name}`;
            
            const caseSummary = document.createElement('p');
            caseSummary.innerHTML = `<strong>Summary:</strong> ${caseItem.Case_Summary}`;
            
            const caseVerdict = document.createElement('p');
            caseVerdict.innerHTML = `<strong>Verdict:</strong> <span class="badge bg-info">${caseItem.Verdict}</span>`;
            
            caseDiv.appendChild(caseTitle);
            caseDiv.appendChild(caseCourt);
            caseDiv.appendChild(caseJudge);
            caseDiv.appendChild(caseSummary);
            caseDiv.appendChild(caseVerdict);
            
            contentDiv.appendChild(caseDiv);
        });
        
        casesDiv.appendChild(contentDiv);
        chatMessages.appendChild(casesDiv);
        
        scrollToBottom();
    }
    
    function addCaseTypeDetailsToChat(details) {
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'message assistant';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content details-content';
        
        const title = document.createElement('h5');
        title.textContent = details.description;
        
        contentDiv.appendChild(title);
        
        // Common legal references
        if (details.common_legal_references) {
            const refsTitle = document.createElement('p');
            refsTitle.innerHTML = '<strong>Common Legal References:</strong>';
            contentDiv.appendChild(refsTitle);
            
            const refsList = document.createElement('ul');
            Object.entries(details.common_legal_references).slice(0, 3).forEach(([ref, count]) => {
                const refItem = document.createElement('li');
                refItem.textContent = `${ref} (${count} cases)`;
                refsList.appendChild(refItem);
            });
            contentDiv.appendChild(refsList);
        }
        
        // Common verdicts
        if (details.common_verdicts) {
            const verdictsTitle = document.createElement('p');
            verdictsTitle.innerHTML = '<strong>Common Verdicts:</strong>';
            contentDiv.appendChild(verdictsTitle);
            
            const verdictsList = document.createElement('ul');
            Object.entries(details.common_verdicts).slice(0, 3).forEach(([verdict, count]) => {
                const verdictItem = document.createElement('li');
                verdictItem.textContent = `${verdict} (${count} cases)`;
                verdictsList.appendChild(verdictItem);
            });
            contentDiv.appendChild(verdictsList);
        }
        
        detailsDiv.appendChild(contentDiv);
        chatMessages.appendChild(detailsDiv);
        
        scrollToBottom();
    }
    
    function addLegalReferenceDetailsToChat(details) {
        // Similar to addCaseTypeDetailsToChat but for legal references
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'message assistant';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content details-content';
        
        const title = document.createElement('h5');
        title.textContent = details.description;
        
        contentDiv.appendChild(title);
        
        // Common case types
        if (details.common_case_types) {
            const typesTitle = document.createElement('p');
            typesTitle.innerHTML = '<strong>Common Case Types:</strong>';
            contentDiv.appendChild(typesTitle);
            
            const typesList = document.createElement('ul');
            Object.entries(details.common_case_types).slice(0, 3).forEach(([type, count]) => {
                const typeItem = document.createElement('li');
                typeItem.textContent = `${type} (${count} cases)`;
                typesList.appendChild(typeItem);
            });
            contentDiv.appendChild(typesList);
        }
        
        // Common verdicts
        if (details.common_verdicts) {
            const verdictsTitle = document.createElement('p');
            verdictsTitle.innerHTML = '<strong>Common Verdicts:</strong>';
            contentDiv.appendChild(verdictsTitle);
            
            const verdictsList = document.createElement('ul');
            Object.entries(details.common_verdicts).slice(0, 3).forEach(([verdict, count]) => {
                const verdictItem = document.createElement('li');
                verdictItem.textContent = `${verdict} (${count} cases)`;
                verdictsList.appendChild(verdictItem);
            });
            contentDiv.appendChild(verdictsList);
        }
        
        detailsDiv.appendChild(contentDiv);
        chatMessages.appendChild(detailsDiv);
        
        scrollToBottom();
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Search functionality
    const searchQuery = document.getElementById('search-query');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    
    searchButton.addEventListener('click', function() {
        const query = searchQuery.value.trim();
        if (query === '') return;
        
        searchResults.innerHTML = '<p>Searching...</p>';
        
        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            searchResults.innerHTML = '';
            
            if (data.cases && data.cases.length > 0) {
                data.cases.forEach(caseItem => {
                    const caseCard = document.createElement('div');
                    caseCard.className = 'case-card';
                    
                    const caseTitle = document.createElement('h4');
                    caseTitle.textContent = `Case ${caseItem.Case_ID}: ${caseItem.Case_Type}`;
                    
                    const caseCourt = document.createElement('p');
                    caseCourt.innerHTML = `<strong>Court:</strong> ${caseItem.Court}`;
                    
                    const caseJudge = document.createElement('p');
                    caseJudge.innerHTML = `<strong>Judge:</strong> ${caseItem.Judge_Name}`;
                    
                    const caseSummary = document.createElement('p');
                    caseSummary.innerHTML = `<strong>Summary:</strong> ${caseItem.Case_Summary}`;
                    
                    const caseVerdict = document.createElement('p');
                    caseVerdict.innerHTML = `<strong>Verdict:</strong> <span class="badge bg-info">${caseItem.Verdict}</span>`;
                    
                    caseCard.appendChild(caseTitle);
                    caseCard.appendChild(caseCourt);
                    caseCard.appendChild(caseJudge);
                    caseCard.appendChild(caseSummary);
                    caseCard.appendChild(caseVerdict);
                    
                    searchResults.appendChild(caseCard);
                });
            } else {
                searchResults.innerHTML = '<p>No matching cases found.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            searchResults.innerHTML = '<p>Error searching for cases.</p>';
        });
    });
    
    // Prediction functionality
    const predictionQuery = document.getElementById('prediction-query');
    const predictionButton = document.getElementById('prediction-button');
    const predictionResults = document.getElementById('prediction-results');
    
    predictionButton.addEventListener('click', function() {
        const description = predictionQuery.value.trim();
        if (description === '') return;
        
        predictionResults.innerHTML = '<p>Analyzing...</p>';
        
        fetch('/case_prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ description: description })
        })
        .then(response => response.json())
        .then(data => {
            predictionResults.innerHTML = '';
            
            const predictionCard = document.createElement('div');
            predictionCard.className = 'prediction-card';
            
            // Case Type Prediction
            const caseTypeItem = document.createElement('div');
            caseTypeItem.className = 'prediction-item';
            
            const caseTypeTitle = document.createElement('h4');
            caseTypeTitle.textContent = 'Case Type Prediction';
            
            const caseTypeBadge = document.createElement('span');
            caseTypeBadge.className = 'badge bg-primary';
            caseTypeBadge.textContent = data.case_type;
            
            caseTypeItem.appendChild(caseTypeTitle);
            caseTypeItem.appendChild(caseTypeBadge);
            
            // Verdict Prediction
            const verdictItem = document.createElement('div');
            verdictItem.className = 'prediction-item';
            
            const verdictTitle = document.createElement('h4');
            verdictTitle.textContent = 'Verdict Prediction';
            
            const verdictBadge = document.createElement('span');
            verdictBadge.className = 'badge bg-success';
            verdictBadge.textContent = data.verdict;
            
            verdictItem.appendChild(verdictTitle);
            verdictItem.appendChild(verdictBadge);
            
            // Legal Reference Prediction
            const referenceItem = document.createElement('div');
            referenceItem.className = 'prediction-item';
            
            const referenceTitle = document.createElement('h4');
            referenceTitle.textContent = 'Legal Reference Prediction';
            
            const referenceBadge = document.createElement('span');
            referenceBadge.className = 'badge bg-info';
            referenceBadge.textContent = data.legal_reference;
            
            referenceItem.appendChild(referenceTitle);
            referenceItem.appendChild(referenceBadge);
            
            // Add all items to card
            predictionCard.appendChild(caseTypeItem);
            predictionCard.appendChild(verdictItem);
            predictionCard.appendChild(referenceItem);
            
            predictionResults.appendChild(predictionCard);
        })
        .catch(error => {
            console.error('Error:', error);
            predictionResults.innerHTML = '<p>Error making predictions.</p>';
        });
    });
    
    // Analytics functionality
    function loadAnalyticsData() {
        fetch('/get_analytics_data')
            .then(response => response.json())
            .then(data => {
                createCaseTypesChart(data.case_types);
                createCourtsChart(data.courts);
                createVerdictsChart(data.verdicts);
                createAppealsChart(data.appeals);
                createTimeChart(data.years);
            })
            .catch(error => {
                console.error('Error:', error);
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
}); 