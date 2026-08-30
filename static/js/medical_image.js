document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("medical-image");
    const analyzeButton = document.getElementById("analyze-image");
    const encryptButton = document.getElementById("encrypt-image");
    const decryptButton = document.getElementById("decrypt-image");
    const selectedFile = document.getElementById("selected-file");
    const analysisResult = document.getElementById("analysis-result");
    const operationChoice = document.getElementById("operation-choice");
    const chooseEncrypt = document.getElementById("choose-encrypt");
    const chooseDecrypt = document.getElementById("choose-decrypt");
    const encryptionControls = document.getElementById("encryption-controls");
    const decryptionPanel = document.getElementById("decryption-panel");
    const decryptionMessage = document.getElementById("decryption-message");
    const imageContainer = document.getElementById("decrypted-image-container");
    const decryptedImage = document.getElementById("decrypted-image");
    const keyInput = document.getElementById("quantum-key-input");
    const eveCheckbox = document.getElementById("eve-checkbox");
    const copyKeyButton = document.getElementById("copy-key");
    let generatedKey = "";
    let currentStage = 0;

    const setText = (id, value) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    };

    const formatBits = (bits) => bits ? bits.join("") : "--";

    const show = (element) => { if (element) element.style.display = "block"; };
    const hide = (element) => { if (element) element.style.display = "none"; };

    const message = (element, text, error = false) => {
        if (!element) return;
        element.textContent = text;
        element.className = error ? "image-message error" : "image-message success";
        show(element);
    };

    const setWorkflow = (step) => {
        ["upload", "bb84", "encrypt"].forEach((name, index) => {
            const element = document.getElementById(`workflow-${name}`);
            if (element) element.classList.toggle("active", index <= step);
        });
    };

    const setStage = (stage) => {
        const stageNames = ["upload", "encryption", "decryption"];
        currentStage = stageNames.indexOf(stage);
        document.querySelectorAll(".stage-panel").forEach((panel) => {
            panel.classList.toggle("stage-visible", panel.classList.contains(`stage-${stage}`));
        });
        setWorkflow(currentStage);
    };

    async function postJson(url, body) {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.message || "Request failed.");
        return result;
    }

    const aiContainer = document.getElementById("medical-ai-analysis-container");
    const pdfViewContainer = document.getElementById("pdf-view-container");
    const pdfDownloadLink = document.getElementById("pdf-download-link");

    async function verifyAndFilterUploadedFile() {
        if (!fileInput || !fileInput.files.length) return;

        const file = fileInput.files[0];
        if (analyzeButton) {
            analyzeButton.disabled = true;
            analyzeButton.textContent = "Verifying Medical Content...";
        }
        setText("bb84-status", "VERIFYING");
        message(analysisResult, "Verifying uploaded file content at initial upload step...");
        hide(aiContainer);
        hide(operationChoice);
        hide(encryptionControls);
        hide(decryptionPanel);

        try {
            const formData = new FormData();
            formData.append("medical_image", file);
            const response = await fetch("/api/detect-medical-image", { method: "POST", body: formData });
            const result = await response.json();

            if (!response.ok || !result.success || !result.is_medical || result.medical_verified === false) {
                fileInput.value = "";
                const state = result.verification_state || "NON_MEDICAL";
                const isUnclear = state === "UNCLEAR";

                selectedFile.textContent = isUnclear ? "File discarded: Unclear medical content" : "File discarded: Non-medical file";
                setText("bb84-status", isUnclear ? "UNCLEAR" : "REJECTED");

                const alertHeader = isUnclear ? "⚠ UNABLE TO VERIFY MEDICAL CONTENT" : "✕ NON-MEDICAL FILE DETECTED";
                const defaultMsg = isUnclear ? "Unable to verify this file as a medical file. Please upload a clearer medical file." : "Non-medical file detected. Please upload a valid medical file.";

                message(analysisResult, `${alertHeader}\n\nStatus: Upload rejected\nMessage: ${result.message || defaultMsg}`, true);
                hide(operationChoice);
                hide(encryptionControls);
                if (chooseEncrypt) chooseEncrypt.disabled = true;
                return false;
            }

            if (chooseEncrypt) chooseEncrypt.disabled = false;
            const organDisplay = result.organ || result.body_region || "Unable to determine reliably";
            const modalityDisplay = result.modality || "Unable to determine reliably";
            const confidenceDisplay = result.certainty === "high" ? "HIGH" : (result.certainty === "medium" ? "MEDIUM" : "UNCERTAIN");
            const statusMessage = "Ready for quantum encryption";

            let displayMsg = `✓ MEDICAL FILE VERIFIED\n\nFile: ${file.name}\nContent Type: ${result.medical_type || "Medical File"}\nOrgan / Region: ${organDisplay}\nModality: ${modalityDisplay}\nConfidence: ${confidenceDisplay}\n\nStatus: ${statusMessage}`;

            message(analysisResult, displayMsg);
            show(operationChoice);
            hide(encryptionControls);
            setText("bb84-status", "READY");
            setStage("upload");
            return true;



        } catch (error) {
            setText("bb84-status", "ERROR");
            message(analysisResult, error.message || "Failed to verify medical content.", true);
            return false;
        } finally {
            if (analyzeButton) {
                analyzeButton.disabled = false;
                analyzeButton.textContent = "🔍 Re-verify Uploaded File";
            }
        }
    }

    fileInput?.addEventListener("change", async () => {
        const hasFile = fileInput.files && fileInput.files.length > 0;
        generatedKey = "";
        selectedFile.textContent = hasFile ? `Selected: ${fileInput.files[0].name}` : "No file selected";
        hide(encryptionControls);
        hide(decryptionPanel);
        hide(imageContainer);
        hide(analysisResult);
        hide(operationChoice);
        hide(aiContainer);
        hide(pdfViewContainer);
        setStage("upload");

        if (hasFile) {
            await verifyAndFilterUploadedFile();
        }
    });

    analyzeButton?.addEventListener("click", async () => {
        if (!fileInput.files.length) return;
        await verifyAndFilterUploadedFile();
    });


    chooseEncrypt?.addEventListener("click", () => {
        hide(operationChoice);
        if (!generatedKey) show(encryptionControls);
        setStage("encryption");
    });

    chooseDecrypt?.addEventListener("click", () => {
        hide(operationChoice);
        setStage("decryption");
    });

    encryptButton?.addEventListener("click", async () => {
        encryptButton.disabled = true;
        encryptButton.textContent = "Generating BB84 Key...";
        setText("bb84-status", "GENERATING");
        message(analysisResult, "Alice and Bob are generating and sifting quantum bases...");

        try {
            const result = await postJson("/api/encrypt-medical-image", { eve: eveCheckbox.checked });
            generatedKey = result.quantum_key;
            keyInput.value = "";
            setText("image-key", result.quantum_key);
            setText("image-qber", `${Number(result.qber).toFixed(2)}%`);
            setText("image-channel", result.secure ? "SAFE" : "COMPROMISED");
            setText("eve-status", result.eve ? "ACTIVE" : "INACTIVE");
            setText("matching-bases", result.matching_bases);
            setText("discarded-bases", result.discarded_bases);
            setText("final-key-length", `${result.key_length} bits`);
            setText("alice-bits", formatBits(result.alice_bits));
            setText("alice-bases", formatBits(result.alice_bases));
            setText("bob-bases", formatBits(result.bob_bases));
            setText("bob-bits", formatBits(result.bob_bits));
            setText("sifted-key", formatBits(result.sifted_key));
            setText("alice-status", "BITS GENERATED");
            setText("bob-status", "BASES MEASURED");
            setText("bb84-status", result.security_status);
            message(document.getElementById("bb84-process-message"), result.eve ? "Eve intercepted the channel. The key was generated, but the output will be marked corrupted during decryption." : "Matching bases were kept and the remaining bases were discarded. The quantum key is ready.");
            show(document.getElementById("bb84-details"));
            show(decryptionPanel);
            show(copyKeyButton);
            hide(encryptionControls);
            setStage("encryption");
            message(analysisResult, "File encrypted. Provide the generated quantum key to the patient for decryption.");
        } catch (error) {
            setText("bb84-status", "ERROR");
            message(analysisResult, error.message, true);
        } finally {
            encryptButton.disabled = false;
            encryptButton.textContent = "🔐 Encrypt Medical Image / Report";
        }
    });

    decryptButton?.addEventListener("click", async () => {
        decryptButton.disabled = true;
        hide(imageContainer);
        hide(aiContainer);
        hide(pdfViewContainer);
        message(decryptionMessage, "Checking quantum key and decrypting medical payload...");

        try {
            const result = await postJson("/api/decrypt-medical-image", { quantum_key: keyInput.value });

            if (result.is_pdf && result.pdf_url) {
                const pdfIframe = document.getElementById("decrypted-pdf-iframe");
                if (pdfIframe) pdfIframe.src = `${result.pdf_url}#toolbar=1&navpanes=1`;
                if (pdfDownloadLink) pdfDownloadLink.href = result.pdf_url;
                show(pdfViewContainer);
                hide(imageContainer);
            } else if (result.image_url) {
                decryptedImage.src = `${result.image_url}?t=${Date.now()}`;
                show(imageContainer);
            }

            if (result.audio_url) {
                let audioElem = document.getElementById("decrypted-audio-player");
                if (!audioElem) {
                    audioElem = document.createElement("audio");
                    audioElem.id = "decrypted-audio-player";
                    audioElem.controls = true;
                    audioElem.style.width = "100%";
                    audioElem.style.marginTop = "1rem";
                    imageContainer.parentNode.insertBefore(audioElem, imageContainer.nextSibling);
                }
                audioElem.src = `${result.audio_url}?t=${Date.now()}`;
                audioElem.style.display = "block";
            }

            if (result.video_url) {
                let videoElem = document.getElementById("decrypted-video-player");
                if (!videoElem) {
                    videoElem = document.createElement("video");
                    videoElem.id = "decrypted-video-player";
                    videoElem.controls = true;
                    videoElem.style.width = "100%";
                    videoElem.style.borderRadius = "8px";
                    videoElem.style.marginTop = "1rem";
                    imageContainer.parentNode.insertBefore(videoElem, imageContainer.nextSibling);
                }
                videoElem.src = `${result.video_url}?t=${Date.now()}`;
                videoElem.style.display = "block";
            }


            message(decryptionMessage, result.message, !result.decrypted);


            // Populate MedGemma AI Analysis ONLY if decryption was successful and channel is secure
            if (result.decrypted && !result.corrupted && result.ai_analysis) {
                const ai = result.ai_analysis;
                const st1 = ai.stage1_identification || {};
                
                setText("ai-input-type", st1.input_type === "medical_report" ? "Medical Report / Document" : (st1.input_type === "medical_image" ? "Medical Diagnostic Image" : "Medical Content"));
                setText("ai-body-region", st1.body_region || "Unknown / Unable to determine");
                setText("ai-modality", st1.modality || "Diagnostic Image");
                setText("ai-report-type", st1.report_type || "N/A");
                setText("ai-certainty", (st1.certainty || "medium").toUpperCase());

                setText("ai-medical-type", ai.medical_image_report_type || "--");
                setText("ai-medical-finding", ai.medical_finding || "--");
                setText("ai-abnormality-defect", ai.abnormality_defect || "--");
                setText("ai-possible-condition", ai.possible_condition || "--");
                setText("ai-medication-info", ai.medication_info || "--");
                setText("ai-simple-explanation", ai.simple_explanation || "--");
                setText("ai-detailed-explanation", ai.detailed_explanation || "--");
                setText("ai-uncertainty", ai.uncertainty || "--");
                setText("ai-next-steps", ai.recommended_next_steps || "--");

                show(aiContainer);
            }




        } catch (error) {
            message(decryptionMessage, error.message, true);
        } finally {
            decryptButton.disabled = false;
        }
    });

    copyKeyButton?.addEventListener("click", async () => {
        await navigator.clipboard.writeText(generatedKey);
        copyKeyButton.textContent = "✓ Quantum Key Copied";
        setStage("decryption");
    });

    document.querySelectorAll(".workflow-step").forEach((step) => {
        step.addEventListener("click", () => {
            const requestedStage = step.dataset.stage;
            setStage(requestedStage);
        });
    });

    setStage("upload");
});

