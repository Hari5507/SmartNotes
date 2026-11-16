const chooseFile = document.getElementById("chooseFile");
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const fileStatus = document.getElementById("fileStatus");

// Open file picker
chooseFile.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
});

// Also allow clicking the drop area
dropzone.addEventListener("click", () => {
    fileInput.click();
});

// When a file is selected
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
    }
});

// Drag over
dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.style.background = "#eef4ff";
});

// Drag leaves
dropzone.addEventListener("dragleave", () => {
    dropzone.style.background = "#fbfcff";
});

// File dropped
dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.style.background = "#fbfcff";

    if (event.dataTransfer.files.length > 0) {
        uploadFile(event.dataTransfer.files[0]);
    }
});


// Upload file to Flask
async function uploadFile(file) {

    // Check file type
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        fileStatus.textContent = "❌ Please select a PDF file.";
        fileStatus.style.color = "#e52d3b";
        return;
    }

    fileStatus.textContent = "⏳ Uploading and extracting your PDF...";
    fileStatus.style.color = "#53627b";

    const formData = new FormData();
    formData.append("file", file);

    try {

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Upload failed");
        }

        fileStatus.textContent =
            `✅ ${data.filename} uploaded successfully! ` +
            `${data.pages} page(s) processed.`;

        fileStatus.style.color = "#1f8a50";

        const extractedContent = document.getElementById("extractedContent");
const pageCount = document.getElementById("pageCount");

sessionStorage.setItem(
    "smartnotes_text",
    data.text || "No readable text was found in this PDF."
);

sessionStorage.setItem(
    "smartnotes_filename",
    data.filename
);

sessionStorage.setItem(
    "smartnotes_pages",
    data.pages
);

window.location.href = "/document.html";

    } catch (error) {

        console.error(error);

        fileStatus.textContent =
            "❌ Could not connect to SmartNotes backend.";

        fileStatus.style.color = "#e52d3b";
    }
}
/* =========================================
   MOBILE MENU
========================================= */

const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const sidebar = document.querySelector(".sidebar");

if (mobileMenuBtn && sidebar) {

    mobileMenuBtn.addEventListener("click", function () {

        sidebar.classList.toggle("menu-open");

        if (sidebar.classList.contains("menu-open")) {
            mobileMenuBtn.textContent = "✕";
        } else {
            mobileMenuBtn.textContent = "☰";
        }

    });

}
/* =========================================
   AI SUMMARIZE
========================================= */

const summarizeBtn = document.getElementById("summarizeBtn");
const aiOutput = document.getElementById("aiOutput");

if (summarizeBtn) {

    summarizeBtn.addEventListener("click", async function () {

        // Check whether document text exists
        if (!documentText || documentText.trim() === "") {

            aiOutput.innerHTML = `
                <div class="ai-placeholder">
                    <div class="ai-placeholder-icon">⚠️</div>
                    <h3>No document content</h3>
                    <p>Please upload a PDF first.</p>
                </div>
            `;

            return;
        }

        // Show loading state
        summarizeBtn.disabled = true;

        aiOutput.innerHTML = `
            <div class="ai-placeholder">
                <div class="ai-placeholder-icon">✨</div>
                <h3>Creating your summary...</h3>
                <p>AI is reading your document. Please wait.</p>
            </div>
        `;

        try {

            const response = await fetch("/ai/summarize", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: documentText
                })

            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Something went wrong.");
            }

            // Display AI result
            aiOutput.innerHTML = `
                <div class="ai-result">
                    <h3>🧠 Summary</h3>
                    <div class="ai-result-text"></div>
                </div>
            `;

            document.querySelector(".ai-result-text").textContent =
                data.result;

        } catch (error) {

            aiOutput.innerHTML = `
                <div class="ai-placeholder">
                    <div class="ai-placeholder-icon">❌</div>
                    <h3>Unable to create summary</h3>
                    <p>${error.message}</p>
                </div>
            `;

        } finally {

            summarizeBtn.disabled = false;

        }

    });

}