class DatasetUploader {
    constructor(dropZoneId, fileInputId, browseBtnId, fileNameId, uploadUrl) {
        this.dropZone = document.getElementById(dropZoneId);
        this.fileInput = document.getElementById(fileInputId);
        this.browseBtn = document.getElementById(browseBtnId);
        this.fileNameDisplay = document.getElementById(fileNameId);
        this.uploadUrl = uploadUrl;

        this.initEvents();
    }

    initEvents() {
        // Browse button click
        this.browseBtn.addEventListener('click', () => this.fileInput.click());

        // File input change
        this.fileInput.addEventListener('change', () => {
            this.displayFileName();
            this.uploadFile();
        });

        // Drag events
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('border-primary');
        });

        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('border-primary');
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('border-primary');
            if (e.dataTransfer.files.length) {
                this.fileInput.files = e.dataTransfer.files;
                this.displayFileName();
                this.uploadFile();
            }
        });
    }

    displayFileName() {
        this.fileNameDisplay.textContent = this.fileInput.files[0]?.name || '';
    }

    uploadFile() {
        const file = this.fileInput.files[0];
        if (!file) {
            alert("Please select a file first.");
            return;
        }

        const formData = new FormData();
        formData.append('dataset_file', file);

        fetch(this.uploadUrl, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .catch(err => console.error(err));
    }
}

// Initialize after DOM loads
document.addEventListener('DOMContentLoaded', function () {
    new DatasetUploader(
        'drop-zone',      // Drop zone ID
        'file-input',     // File input ID
        'browse-btn',     // Browse button ID
        'file-name',      // File name display ID
        "{{ url_for('views.upload_dataset') }}" // Flask upload endpoint
    );
});