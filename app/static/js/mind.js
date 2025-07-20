class MINDInterface {
    constructor() {
        this.datasetItems = document.querySelectorAll('.dataset-item');
        this.modeSelectors = document.querySelectorAll('.mode-selectors');
        this.form = document.getElementById('mind-number-form');
        this.numberInput = document.getElementById('numberInput');
        this.topicButtons = document.querySelectorAll('.topic_button') ?? [];

        this.fetchMindStatus();
        this.initEventListeners();
        this.initCarouselControls();
    }

    initEventListeners() {
        // Dataset selection
        this.datasetItems.forEach(item => {
            item.addEventListener('click', () => {
                const datasetName = item.textContent.trim();
                this.handleDatasetSelection(datasetName);
            });
        });

        // Form submission
        if (this.form) {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                const value = parseInt(this.numberInput.value, 10);
                if (!isNaN(value)) {
                    this.handleSubmitNumber(value);
                }
            });
        }

        // Mode selectors
        this.modeSelectors.forEach(selector => {
            selector.addEventListener('click', (e) => {
                e.preventDefault();
                const instruction = selector.textContent.trim();
                this.handleInstruction(instruction);

            });
        });
        // Topic buttons
        if (this.topicButtons) {
            console.log("Initializing topic buttons");
            this.initTopicButtons();
        }
    }
        initTopicButtons() {
        this.topicButtons.forEach(button => {
            button.addEventListener('click', () => {
                const topicId = button.dataset.topicId;
                this.handleTopicClick(topicId);
                this.loadTopicDocuments(topicId); 
            });
        });
    }

    initCarouselControls() {
        const carousel = document.getElementById('document-carousel');
        const prevBtn = document.getElementById('prev-doc');
        const nextBtn = document.getElementById('next-doc');
        // const returnBtn = document.getElementById('return-btn');

        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                $('#document-carousel').carousel('prev'); // Bootstrap's carousel function
            });
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                $('#document-carousel').carousel('next');
            });
        }

        // if (returnBtn) {
        //     returnBtn.addEventListener('click', () => {
        //         const topicListCard = document.getElementById('topic-list-card');
        //         const carouselCard = document.getElementById('document-carousel-card');
        //         topicListCard.classList.toggle('d-none');
        //         carouselCard.classList.toggle('d-none');
        //     });
        // }
    }

    loadTopicDocuments(topicId) {
        fetch('/topic_selection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCSRFToken()
            },
            body: JSON.stringify({ topic_id: topicId })
        })
        .then(res => res.json())
        .then(data => {
            console.log("Topic documents data:", data);
            if (data.topic_documents && Array.isArray(data.topic_documents)) {
                this.renderCarousel(data.topic_documents);
            } else {
                console.error("Error loading documents", data);
            }
        })
        .catch(err => console.error("Error fetching topic documents:", err));
    }

    fetchMindStatus() {
        fetch('http://localhost:93/status') 
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to fetch status");
                }
                return response.json();
            })
            .then(data => {
                this.lastInstruction = data.state; 
                console.log("MIND status retrieved:", this.lastInstruction);
            })
            .catch(error => {
                console.error("Error fetching MIND status:", error);
            });
    }


    getCSRFToken() {
        const cookieValue = document.cookie
            .split('; ')
            .find(row => row.startsWith('csrf_token='))
            ?.split('=')[1];
        return cookieValue || '';
    }


    renderCarousel(documents) {
        const carouselContainer = document.getElementById('carousel-inner');
        const topicListCard = document.getElementById('topic-list-card');
        const carouselCard = document.getElementById('document-carousel-card');

        carouselContainer.innerHTML = ''; // Clear previous
        documents.forEach((doc, index) => {
            const activeClass = index === 0 ? 'active' : '';
            carouselContainer.innerHTML += `
                <div class="carousel-item ${activeClass}">
                    <div class="p-3 border rounded">
                        <p><strong>Text:</strong> ${doc.raw_text || 'No content available'}</p>
                    </div>
                </div>
            `;
        });

        // Show carousel, hide topic list
        topicListCard.classList.add('d-none');
        carouselCard.classList.remove('d-none');

        // Re-initialize Bootstrap carousel
        $('#document-carousel').carousel(0);
    }

    handleDatasetSelection(datasetName) {
        console.log("Selected dataset:", datasetName);

        fetch('/detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCSRFToken() // if using Flask-WTF/CSRF
            },
            body: JSON.stringify({ dataset: datasetName })
        })
        .then(res => res.json())
        .then(data => {
            console.log("Backend response:", data);
            // Optionally update UI or notify the user
        })
        .catch(err => console.error("Dataset selection error:", err));

        //wait for the backend to process the dataset selection
        setTimeout(() => {
            window.location.reload();
        }, 1000); // Adjust the timeout as needed

    }

    handleInstruction(instruction) {
        console.log("Selected instruction:", instruction); 
        fetch('/mode_selection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCSRFToken() 
            },
            body: JSON.stringify({ instruction: instruction })
        })
        .then(res => res.json())
        .then(data => {
            console.log("Backend response:", data);
        })
        .catch(err => console.error("Dataset selection error:", err));    
        setTimeout(() => {
            window.location.reload();
        }, 1000); // Adjust the timeout as needed
    }

    handleTopicClick(topicId) {
        console.log("Selected topic:", topicId);
        if (this.lastInstruction === 'topic_exploration') {
            fetch('/topic_selection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken() // if using Flask-WTF/CSRF
                },
                body: JSON.stringify({ topic_id: topicId })
            })
            .then(res => res.json())
            .then(data => {
                console.log("Backend response:", data);
                // Optionally update UI or notify the user
            })
            .catch(err => console.error("Topic selection error:", err));
        }
        else if (this.lastInstruction === 'topic_analysis') {
            fetch('/analyze_topic', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken() // if using Flask-WTF/CSRF
                },
                body: JSON.stringify({ topic_id: topicId })
            })
            .then(res => res.json())
            .then(data => {
                console.log("Backend response:", data);
                // Optionally update UI or notify the user
            })
            .catch(err => console.error("Topic analysis error:", err));
        } else {
            console.error("Invalid instruction for topic selection:", this.lastInstruction);
            return;
        }
        // setTimeout(() => {
        //     window.location.reload();
        // }, 1000); // Adjust the timeout as needed
    }

    handleSubmitNumber(value) {
        console.log("Submitted number:", value);
        // TODO: Send number to backend
        /*
        fetch('/submit-number', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ number: value })
        })
        .then(res => res.json())
        .then(data => console.log(data));
        */
    }


}

// Instantiate when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("MINDInterface initialized");
    const mindInterface = new MINDInterface();
    // const socket = new WebSocket(`ws://${window.location.host}/ws`);
    
});
