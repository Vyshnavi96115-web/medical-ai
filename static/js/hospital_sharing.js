document.addEventListener("DOMContentLoaded", () => {
    const prepareButton = document.getElementById("prepare-button");
    const generateButton = document.getElementById("generate-button");
    const decryptButton = document.getElementById("decrypt-button");
    const copyKeyButton = document.getElementById("copy-key");
    const resetButton = document.getElementById("reset-button");
    const eveCheckbox = document.getElementById("eve-checkbox");
    const quantumKeyInput = document.getElementById("quantum-key");

    const recordError = document.getElementById("record-error");
    const securityStatus = document.getElementById("security-status");
    const bb84Details = document.getElementById("bb84-details");
    const decryptResult = document.getElementById("decrypt-result");
    const recordOutput = document.getElementById("record-output");

    let generatedKey = "";

    const show = (el) => { if (el) el.hidden = false; };
    const hide = (el) => { if (el) el.hidden = true; };
    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    const setStageActive = (stageName) => {
        document.querySelectorAll(".hospital-step").forEach(step => {
            const isActive = step.dataset.stage === stageName;
            step.classList.toggle("active", isActive);
            step.classList.toggle("locked", !isActive);
        });
        document.querySelectorAll(".hospital-stage").forEach(stage => {
            stage.classList.toggle("stage-visible", stage.classList.contains(`stage-${stageName}`));
        });
    };

    prepareButton?.addEventListener("click", async () => {
        const payload = {
            patient_id: document.getElementById("patient-id")?.value || "",
            patient_name: document.getElementById("patient-name")?.value || "",
            medical_record_number: document.getElementById("medical-record-number")?.value || "",
            age: document.getElementById("patient-age")?.value || "",
            problem: document.getElementById("problem")?.value || "",
            diagnosis: document.getElementById("diagnosis")?.value || "",
            doctor: document.getElementById("doctor")?.value || "",
            department: document.getElementById("department")?.value || "",
            treatment: document.getElementById("treatment")?.value || "",
            patient_summary: document.getElementById("patient-summary")?.value || ""
        };

        const imageInput = document.getElementById("medical-image");
        const formData = new FormData();
        Object.keys(payload).forEach(k => formData.append(k, payload[k]));
        if (imageInput && imageInput.files.length > 0) {
            formData.append("medical_image", imageInput.files[0]);
        }

        try {
            prepareButton.disabled = true;
            const resp = await fetch("/api/hospital-sharing/prepare", { method: "POST", body: formData });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.message || "Failed to prepare data.");

            hide(recordError);
            setStageActive("security");
        } catch (err) {
            if (recordError) {
                recordError.textContent = err.message;
                show(recordError);
            }
        } finally {
            prepareButton.disabled = false;
        }
    });

    generateButton?.addEventListener("click", async () => {
        try {
            generateButton.disabled = true;
            generateButton.textContent = "Generating BB84 Key...";

            const eve = eveCheckbox ? eveCheckbox.checked : False;
            const bb84Resp = await fetch("/api/hospital-sharing/bb84", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ eve })
            });
            const bb84Data = await bb84Resp.json();
            if (!bb84Resp.ok || !bb84Data.success) throw new Error(bb84Data.message || "BB84 Key Generation failed.");

            const encResp = await fetch("/api/hospital-sharing/encrypt", { method: "POST" });
            const encData = await encResp.json();

            generatedKey = bb84Data.quantum_key;
            if (copyKeyButton) copyKeyButton.disabled = false;
            if (decryptButton) decryptButton.disabled = false;

            setText("alice-bits", bb84Data.bb84.alice_bits.join(""));
            setText("alice-bases", bb84Data.bb84.alice_bases.join(""));
            setText("bob-bases", bb84Data.bb84.bob_bases.join(""));
            setText("bob-bits", bb84Data.bb84.bob_bits.join(""));
            setText("sifted-key", bb84Data.bb84.sifted_key.join(""));
            setText("matching-bits", bb84Data.bb84.compared_bits);
            setText("discarded-bits", bb84Data.bb84.total_bits - bb84Data.bb84.compared_bits);
            setText("total-bits", bb84Data.bb84.total_bits);
            setText("sifted-length", bb84Data.bb84.sifted_key.length);
            setText("compared-bits", bb84Data.bb84.compared_bits);
            setText("incorrect-bits", bb84Data.bb84.incorrect_bits);
            setText("qber", `${bb84Data.qber}%`);
            setText("qber-decision", bb84Data.secure ? "SECURE CHANNEL (QBER < 11%)" : "COMPROMISED CHANNEL (EAVESDROPPING DETECTED)");
            setText("final-key-length", `${bb84Data.key_length} BITS`);
            setText("final-key", bb84Data.quantum_key);

            if (eveCheckbox && eveCheckbox.checked) {
                setText("eve-status", "ACTIVE INTERCEPTION");
                setText("eve-bases", bb84Data.bb84.eve_bases.join(""));
                setText("eve-bits", bb84Data.bb84.eve_bits.join(""));
            } else {
                setText("eve-status", "INACTIVE");
                setText("eve-bases", "No interception performed.");
                setText("eve-bits", "");
            }

            show(bb84Details);
            if (securityStatus) {
                securityStatus.textContent = encData.message;
                show(securityStatus);
            }
        } catch (err) {
            if (securityStatus) {
                securityStatus.textContent = err.message;
                show(securityStatus);
            }
        } finally {
            generateButton.disabled = false;
            generateButton.textContent = "🔐 ENCRYPT & GENERATE BB84 KEY";
        }
    });

    copyKeyButton?.addEventListener("click", async () => {
        if (!generatedKey) return;
        await navigator.clipboard.writeText(generatedKey);
        if (quantumKeyInput) quantumKeyInput.value = generatedKey;
        copyKeyButton.textContent = "✓ QUANTUM KEY COPIED";
        setStageActive("decryption");
    });

    decryptButton?.addEventListener("click", async () => {
        const enteredKey = quantumKeyInput ? quantumKeyInput.value.trim() : "";
        try {
            decryptButton.disabled = true;
            const resp = await fetch("/api/hospital-sharing/decrypt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ quantum_key: enteredKey })
            });
            const data = await resp.json();

            if (decryptResult) {
                decryptResult.textContent = data.message;
                show(decryptResult);
            }

            if (data.record) {
                const dl = document.getElementById("output-fields");
                if (dl) {
                    dl.innerHTML = "";
                    Object.keys(data.record).forEach(k => {
                        if (k === "medical_image") return;
                        const dt = document.createElement("dt");
                        dt.textContent = k.replace(/_/g, " ").toUpperCase();
                        const dd = document.createElement("dd");
                        dd.textContent = data.record[k];
                        dl.appendChild(dt);
                    });
                }
                show(recordOutput);
            }
        } catch (err) {
            if (decryptResult) {
                decryptResult.textContent = err.message;
                show(decryptResult);
            }
        } finally {
            decryptButton.disabled = false;
        }
    });

    resetButton?.addEventListener("click", async () => {
        await fetch("/api/hospital-sharing/reset", { method: "POST" });
        location.reload();
    });

    document.querySelectorAll(".hospital-step").forEach(step => {
        step.addEventListener("click", () => {
            setStageActive(step.dataset.stage);
        });
    });
});
