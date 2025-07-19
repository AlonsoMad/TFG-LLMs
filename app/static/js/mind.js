class MINDInterface {
    constructor() {
        this.datasetItems = document.querySelectorAll('.dataset-item');
        this.modeSelectors = document.querySelectorAll('.mode-selectors');
        this.form = document.getElementById('mind-number-form');
        this.numberInput = document.getElementById('numberInput');
        this.navLinks = document.querySelectorAll('.control');

        this.initEventListeners();
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



        // Navigation links
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const target = link.getAttribute('href');
                this.handleNavigation(target);
            });
        });
    }

    getCSRFToken() {
        const cookieValue = document.cookie
            .split('; ')
            .find(row => row.startsWith('csrf_token='))
            ?.split('=')[1];
        return cookieValue || '';
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
                'X-CSRFToken': this.getCSRFToken() // if using Flask-WTF/CSRF
            },
            body: JSON.stringify({ instruction: instruction })
        })
        .then(res => res.json())
        .then(data => {
            console.log("Backend response:", data);
            // Optionally update UI or notify the user
        })
        .catch(err => console.error("Dataset selection error:", err));    
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

    handleNavigation(targetId) {
        console.log("Navigating to:", targetId);
        // You can enhance this if you want to do more than just toggle tabs
    }
}

// Instantiate when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("MINDInterface initialized");
    const mindInterface = new MINDInterface();
    // const socket = new WebSocket(`ws://${window.location.host}/ws`);
    
});
