document.addEventListener("DOMContentLoaded", () => {
    const stages = ["prepare", "security", "access"];
    let unlockedStage = 0;
    let currentKey = "";
    const get = (id) => document.getElementById(id);
    const setText = (id, value) => { if (get(id)) get(id).textContent = value ?? "--"; };
    const showMessage = (id, value, error = false) => { const element = get(id); element.hidden = false; element.textContent = value; element.className = `cloud-message${error ? " error" : ""}`; };
    const bits = (values) => values && values.length ? values.join(" ") : "--";
    const setStage = (stage) => {
        const stageIndex = stages.indexOf(stage);
        if (stageIndex < 0 || stageIndex > unlockedStage) { showMessage("workflow-message", "Complete the previous stage first.", true); return; }
        document.querySelectorAll(".cloud-stage").forEach((panel) => panel.classList.toggle("stage-visible", panel.classList.contains(`stage-${stage}`)));
        document.querySelectorAll(".cloud-step").forEach((step, index) => { step.classList.toggle("active", index === stageIndex); step.classList.toggle("completed", index < unlockedStage); });
    };
    const updateStats = (stats) => Object.entries(stats || {}).forEach(([key, value]) => setText(key.replaceAll("_", "-"), value));
    const postJson = async (url, body) => { const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); const result = await response.json(); return { response, result }; };

    get("prepare-button").addEventListener("click", async () => {
        const fieldIds = ["patient-name", "patient-id", "age", "medical-condition", "diagnosis", "treatment", "doctor-notes", "heart-rate", "spo2", "blood-pressure", "temperature", "respiration-rate"];
        if (fieldIds.some((id) => !get(id).value.trim())) { showMessage("prepare-message", "Complete all medical record and vital-sign fields.", true); return; }
        const formData = new FormData();
        fieldIds.forEach((id) => formData.append(id.replaceAll("-", "_"), get(id).value));
        if (get("medical-image").files[0]) formData.append("medical_image", get("medical-image").files[0]);
        if (get("medical-report").files[0]) formData.append("medical_report", get("medical-report").files[0]);
        get("prepare-button").disabled = true;
        try { const response = await fetch("/api/healthcare-cloud/prepare", { method: "POST", body: formData }); const result = await response.json(); if (!response.ok || !result.success) throw new Error(result.message); unlockedStage = 1; showMessage("workflow-message", result.message); setStage("security"); }
        catch (error) { showMessage("prepare-message", error.message, true); }
        finally { get("prepare-button").disabled = false; }
    });

    get("encrypt-button").addEventListener("click", async () => {
        get("encrypt-button").disabled = true; showMessage("security-message", "Running the existing BB84 implementation and storing the encrypted record...");
        try {
            const { response, result } = await postJson("/api/healthcare-cloud/encrypt", { eve: get("eve-checkbox").checked });
            if (!response.ok) throw new Error(result.message || "Cloud encryption failed.");
            const data = result.bb84; currentKey = result.quantum_key;
            setText("alice-bits", bits(data.alice_bits)); setText("alice-bases", bits(data.alice_bases)); setText("bob-bases", bits(data.bob_bases)); setText("bob-bits", bits(data.bob_bits));
            setText("eve-status", result.eve ? "ACTIVE" : "OFF"); setText("eve-bases", result.eve ? bits(data.eve_bases) : "Eve: OFF"); setText("eve-bits", result.eve ? bits(data.eve_bits) : "");
            setText("basis-comparison", data.alice_bases.map((aliceBasis, index) => `${aliceBasis}                 ${data.bob_bases[index]}                 ${data.matching_positions[index]}`).join("\n"));
            setText("total-bits", data.total_bits); setText("matching-bits", data.compared_bits); setText("discarded-bits", data.total_bits - data.compared_bits); setText("sifted-key", bits(data.sifted_key)); setText("sifted-length", data.sifted_key.length);
            setText("qber", `${Number(result.qber).toFixed(2)}%`); setText("stored-qber", `${Number(result.qber).toFixed(2)}%`); setText("security-status", result.secure ? "SECURE" : "INSECURE"); setText("qber-warning", result.secure ? "QUANTUM CHANNEL SECURE" : "EAVESDROPPING DETECTED · QUANTUM CHANNEL INSECURE · DATA INTEGRITY COMPROMISED");
            setText("key-length", `Key Length: ${result.key_length} bits`); setText("quantum-key", currentKey); get("bb84-details").hidden = false; get("copy-key").disabled = false; setText("record-status", "ENCRYPTED"); updateStats(result.stats); unlockedStage = 2; setStage("access");
            showMessage("security-message", result.secure ? "MEDICAL RECORD ENCRYPTED · STORED IN QUANTUM-SECURE CLOUD" : "CLOUD RECORD SECURITY COMPROMISED · DATA INTEGRITY COMPROMISED", !result.secure);
        } catch (error) { showMessage("security-message", error.message, true); }
        finally { get("encrypt-button").disabled = false; }
    });

    get("copy-key").addEventListener("click", async () => { try { await navigator.clipboard.writeText(currentKey); } catch (error) { const area = document.createElement("textarea"); area.value = currentKey; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove(); } showMessage("workflow-message", "Quantum key copied successfully."); });
    get("access-key").addEventListener("input", () => { get("access-button").disabled = !get("access-key").value.trim(); });
    get("access-button").addEventListener("click", async () => {
        get("access-button").disabled = true; get("record-output").hidden = true;
        try {
            const { response, result } = await postJson("/api/healthcare-cloud/access", { quantum_key: get("access-key").value });
            if (response.status === 401) throw new Error("ACCESS DENIED · DECRYPTION FAILED · INVALID QUANTUM KEY");
            if (!result.record) throw new Error(result.message || "Cloud record could not be accessed.");
            const record = result.record; const compromised = Boolean(result.compromised); const labels = { "Patient Name": record.patient_name, "Patient ID": record.patient_id, Age: record.age, "Medical Condition": record.medical_condition, Diagnosis: record.diagnosis, Treatment: record.treatment, "Doctor Notes": record.doctor_notes, "Heart Rate": record.heart_rate, SpO2: record.spo2, "Blood Pressure": record.blood_pressure, Temperature: record.temperature, "Respiration Rate": record.respiration_rate };
            get("output-fields").replaceChildren(...Object.entries(labels).flatMap(([label, value]) => { const term = document.createElement("dt"); term.textContent = label; const description = document.createElement("dd"); description.textContent = value || "--"; return [term, description]; }));
            get("download-area").replaceChildren(); ["medical_image", "medical_report"].forEach((field) => { if (!record[field]) return; const link = document.createElement("a"); link.className = "download-link"; link.textContent = `OPEN ${field === "medical_image" ? "MEDICAL IMAGE" : "MEDICAL REPORT"}`; link.href = `data:${record[field].mime_type};base64,${record[field].data}`; link.download = record[field].filename || field; get("download-area").append(link); });
            setText("access-title", compromised ? "CORRUPTED / UNTRUSTED DATA" : "AUTHORIZED ACCESS"); setText("access-warning", compromised ? "DATA INTEGRITY COMPROMISED · QUANTUM CHANNEL INSECURE · CLOUD RECORD CANNOT BE TRUSTED" : "CLOUD RECORD DECRYPTED · DATA INTEGRITY VERIFIED"); get("access-warning").className = compromised ? "compromised" : ""; get("record-output").hidden = false; showMessage("access-message", result.message, compromised);
        } catch (error) { showMessage("access-message", error.message, true); }
        finally { get("access-button").disabled = false; }
    });
    get("access-key").addEventListener("keydown", (event) => { if (event.key === "Enter" && !get("access-button").disabled) get("access-button").click(); });
    document.querySelectorAll(".cloud-step").forEach((step) => step.addEventListener("click", () => setStage(step.dataset.stage))); setStage("prepare");
});
