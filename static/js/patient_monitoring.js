document.addEventListener("DOMContentLoaded", () => {
    const patientId = document.getElementById("patient-id");
    const patientName = document.getElementById("patient-name");
    const monitorButton = document.getElementById("monitor-button");
    const proceedButton = document.getElementById("proceed-button");
    const secureButton = document.getElementById("secure-button");
    const eveCheckbox = document.getElementById("eve-checkbox");
    const receiverSection = document.getElementById("receiver-section");
    const receiverKey = document.getElementById("receiver-key");
    const decryptButton = document.getElementById("decrypt-button");
    const copyKeyButton = document.getElementById("copy-key");
    const securityMessage = document.getElementById("security-message");
    const decryptMessage = document.getElementById("decrypt-message");
    const patientOutput = document.getElementById("patient-output");
    let monitoring = false;
    let pollTimer = null;
    let receiverPollTimer = null;
    const heartRateHistory = [];
    const chart = document.getElementById("heart-rate-chart");
    const securityStage = document.getElementById("security-stage");
    const bb84Details = document.getElementById("bb84-details");
    let currentStage = 0;
    let unlockedStage = 0;
    let generatedKey = "";

    const setWorkflow = (activeStep) => {
        document.querySelectorAll(".monitor-step").forEach((step, index) => {
            step.classList.toggle("active", index <= activeStep);
            step.classList.toggle("locked", index > activeStep);
        });
        setText("workflow-state-one", activeStep >= 1 ? "✓ Completed" : "● Active");
        setText("workflow-state-two", activeStep >= 2 ? "✓ Completed" : activeStep === 1 ? "● Active" : "🔒 Locked");
        setText("workflow-state-three", activeStep >= 3 ? "✓ Completed" : activeStep === 2 ? "● Active" : "🔒 Locked");
    };

    const setStage = (stage) => {
        const stages = ["stage-one", "stage-two", "stage-three"];
        const requestedStage = stages.indexOf(stage);
        if (requestedStage < 0 || requestedStage > unlockedStage) return;
        currentStage = requestedStage;
        document.querySelectorAll(".stage-panel").forEach((panel) => {
            panel.classList.toggle("stage-visible", panel.classList.contains(stage));
        });
        setWorkflow(currentStage);
    };

    const setText = (id, value) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    };

    const formatBits = (bits) => bits && bits.length ? bits.join(" ") : "--";

    const showMessage = (element, text, error = false) => {
        element.hidden = false;
        element.textContent = text;
        element.className = error ? "image-message error" : "image-message success";
    };

    const updateVitals = (vitals) => {
        setText("heart-rate", vitals.heart_rate);
        setText("spo2", vitals.spo2);
        setText("blood-pressure", vitals.blood_pressure);
        setText("temperature", Number(vitals.temperature).toFixed(1));
        setText("respiration-rate", vitals.respiration_rate);
        heartRateHistory.push(vitals.heart_rate);
        if (heartRateHistory.length > 30) heartRateHistory.shift();
        drawHeartRateChart();
    };

    const updateReceiverVitals = (packet) => {
        setText("output-name", packet.patient_name);
        setText("output-id", packet.patient_id);
        setText("output-heart-rate", packet.heart_rate);
        setText("output-spo2", packet.spo2);
        setText("output-blood-pressure", packet.blood_pressure);
        setText("output-temperature", Number(packet.temperature).toFixed(1));
        setText("output-respiration", packet.respiration_rate);
        setText("last-updated", packet.timestamp ? new Date(packet.timestamp * 1000).toLocaleTimeString() : "--");
    };

    const drawHeartRateChart = () => {
        if (!chart) return;
        const context = chart.getContext("2d");
        const width = chart.clientWidth;
        const height = chart.clientHeight;
        const scale = window.devicePixelRatio || 1;
        chart.width = width * scale;
        chart.height = height * scale;
        context.scale(scale, scale);
        context.clearRect(0, 0, width, height);
        if (heartRateHistory.length < 2) return;
        const minimum = 50;
        const maximum = 115;
        context.strokeStyle = "#15acd8";
        context.lineWidth = 2;
        context.beginPath();
        heartRateHistory.forEach((value, index) => {
            const x = (index / (heartRateHistory.length - 1)) * width;
            const y = height - ((value - minimum) / (maximum - minimum)) * height;
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.stroke();
    };

    receiverKey.addEventListener("input", () => {
        decryptButton.disabled = receiverSection.hidden || !receiverKey.value.trim();
    });

    async function fetchVitals() {
        const response = await fetch("/api/patient-monitoring/vitals");
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.message || "Vitals unavailable.");
        updateVitals(result.vitals);
    }

    monitorButton.addEventListener("click", async () => {
        if (monitoring) {
            monitoring = false;
            clearInterval(pollTimer);
            monitorButton.textContent = "▶ START MONITORING";
            secureButton.disabled = true;
            proceedButton.disabled = true;
            setText("monitor-status", "MONITORING STOPPED");
            setStage("stage-one");
            return;
        }

        monitorButton.disabled = true;
        try {
            const response = await fetch("/api/patient-monitoring/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patient_id: patientId.value, patient_name: patientName.value })
            });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.message || "Monitoring could not start.");
            monitoring = true;
            updateVitals(result.vitals);
            pollTimer = setInterval(fetchVitals, 1500);
            monitorButton.textContent = "■ STOP MONITORING";
            secureButton.disabled = false;
            proceedButton.disabled = false;
            setText("monitor-status", "LIVE MONITORING");
            securityStage.classList.add("unlocked");
            bb84Details.hidden = true;
            unlockedStage = 1;
            setWorkflow(0);
        } catch (error) {
            showMessage(securityMessage, error.message, true);
        } finally {
            monitorButton.disabled = false;
        }
    });

    proceedButton.addEventListener("click", () => {
        bb84Details.hidden = true;
        secureButton.disabled = false;
        eveCheckbox.disabled = false;
        securityStage.classList.add("unlocked");
        setStage("stage-two");
    });

    secureButton.addEventListener("click", async () => {
        secureButton.disabled = true;
        setText("bb84-status", "GENERATING");
        showMessage(securityMessage, "Creating patient-data packet and running BB84...");
        try {
            const response = await fetch("/api/patient-monitoring/secure", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patient_id: patientId.value, patient_name: patientName.value, eve: eveCheckbox.checked })
            });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.message || "Security session failed.");
            generatedKey = result.quantum_key;
            setText("bb84-status", result.secure ? "KEY GENERATED" : "WARNING");
            setText("qber", `${Number(result.qber).toFixed(2)}%`);
            setText("quantum-key", `${result.key_length} bits`);
            setText("channel-status", result.channel_status);
            setText("eve-status", result.eve ? "ACTIVE" : "INACTIVE");
            setText("quantum-key", `${result.key_length} bits`);
            showMessage(securityMessage, result.secure ? "✓ QUANTUM CHANNEL SECURE. Patient data has been encrypted successfully." : "⚠ QUANTUM CHANNEL COMPROMISED. Patient data transmission cannot be trusted.", !result.secure);
            setText("encryption-status", result.secure ? "ENCRYPTION STATUS · ✓ ENCRYPTED" : "ENCRYPTION STATUS · NOT TRUSTED");
            const bb84 = result.bb84;
            setText("alice-bits", formatBits(bb84.alice_bits));
            setText("alice-bases", formatBits(bb84.alice_bases));
            setText("bob-bases", formatBits(bb84.bob_bases));
            setText("bob-bits", formatBits(bb84.bob_bits));
            setText("eve-bases", formatBits(bb84.eve_bases));
            setText("eve-bits", formatBits(bb84.eve_bits));
            setText("matching-positions", formatBits(bb84.matching_positions));
            setText("total-bits", 128);
            setText("matching-bits", bb84.compared_bits);
            setText("discarded-bits", 128 - bb84.compared_bits);
            setText("sifted-key", formatBits(bb84.sifted_key));
            setText("compared-bits", bb84.compared_bits);
            setText("incorrect-bits", bb84.incorrect_bits);
            setText("qber-decision", result.secure ? "✓ QUANTUM CHANNEL SECURE" : "⚠ QUANTUM CHANNEL COMPROMISED");
            setText("final-key-length", `${result.key_length} bits`);
            setText("final-key", result.quantum_key);
            setText("eve-detail-status", result.eve ? "ACTIVE" : "INACTIVE");
            document.getElementById("bb84-details").hidden = false;
            setWorkflow(1);
            receiverSection.hidden = !result.secure;
            decryptButton.disabled = !result.secure;
                unlockedStage = 2;
                receiverSection.hidden = false;
                decryptButton.disabled = false;
                setWorkflow(1);
        } catch (error) {
            showMessage(securityMessage, error.message, true);
        } finally {
            secureButton.disabled = false;
        }
    });

    copyKeyButton.addEventListener("click", async () => {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(generatedKey);
            } else {
                const fallback = document.createElement("textarea");
                fallback.value = generatedKey;
                fallback.setAttribute("readonly", "");
                fallback.style.position = "fixed";
                fallback.style.opacity = "0";
                document.body.appendChild(fallback);
                fallback.select();
                document.execCommand("copy");
                fallback.remove();
            }
            showMessage(decryptMessage, "Quantum key copied successfully. Enter it below to decrypt the live data.");
        } catch (error) {
            showMessage(decryptMessage, "Quantum key is ready. Enter it manually in the receiver field.", true);
        }
        setStage("stage-three");
    });

    decryptButton.addEventListener("click", async () => {
        decryptButton.disabled = true;
        patientOutput.hidden = true;
        showMessage(decryptMessage, "Authenticating quantum key...");
        try {
            const response = await fetch("/api/patient-monitoring/decrypt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ quantum_key: receiverKey.value })
            });
            const result = await response.json();
            if (!result.success && !result.compromised) throw new Error(result.message || "Decryption failed.");
            const packet = result.packet;
            if (result.compromised) {
                setText("output-name", packet.patient_name);
                setText("output-id", packet.patient_id);
                setText("output-heart-rate", packet.heart_rate);
                setText("output-spo2", packet.spo2);
                setText("output-blood-pressure", packet.blood_pressure);
                setText("output-temperature", packet.temperature);
                setText("output-respiration", packet.respiration_rate);
                patientOutput.hidden = false;
                showMessage(decryptMessage, result.message, true);
                return;
            }
            setText("output-name", packet.patient_name);
            setText("output-id", packet.patient_id);
            setText("output-heart-rate", packet.heart_rate);
            setText("output-spo2", packet.spo2);
            setText("output-blood-pressure", packet.blood_pressure);
            setText("output-temperature", packet.temperature);
            setText("output-respiration", packet.respiration_rate);
            patientOutput.hidden = false;
            showMessage(decryptMessage, "✓ DECRYPTION SUCCESSFUL");
            updateReceiverVitals(packet);
            clearInterval(receiverPollTimer);
            receiverPollTimer = setInterval(async () => {
                try {
                    const streamResponse = await fetch("/api/patient-monitoring/decrypt", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ quantum_key: receiverKey.value })
                    });
                    const streamResult = await streamResponse.json();
                    if (streamResponse.ok && streamResult.decrypted) updateReceiverVitals(streamResult.packet);
                } catch (error) {
                    clearInterval(receiverPollTimer);
                }
            }, 1000);
                setWorkflow(3);
        } catch (error) {
            showMessage(decryptMessage, `❌ DECRYPTION FAILED: ${error.message}`, true);
        } finally {
            decryptButton.disabled = false;
        }
    });

    document.querySelectorAll(".monitor-step").forEach((step) => {
        step.addEventListener("click", () => {
            const requestedStage = step.dataset.stage;
            const requestedIndex = ["stage-one", "stage-two", "stage-three"].indexOf(requestedStage);
            if (requestedIndex <= unlockedStage) setStage(requestedStage);
        });
    });

    setStage("stage-one");
});
