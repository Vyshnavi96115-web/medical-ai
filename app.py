from flask import Flask, render_template, jsonify, request
from bb84 import BB84
from medical_detector import MedicalImageDetector
from medical_encryption import MedicalImageEncryptor
from medical_decryption import MedicalImageDecryptor
from patient_monitor import VitalSignGenerator
from medical_ai import MedicalContentValidator, MedGemmaAnalyzer

import hmac
import base64
import io
import json
import os
import random
import time
import uuid
from PIL import Image, ImageDraw
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# COMPONENTS
# ============================================================

detector = MedicalImageDetector()

encryptor = MedicalImageEncryptor()

decryptor = MedicalImageDecryptor()

validator = MedicalContentValidator()

medgemma_analyzer = MedGemmaAnalyzer()



# ============================================================
# CURRENT MEDICAL SECURITY SESSION
# ============================================================
#
# For this project demonstration, one active medical
# security session is maintained.
#
# Later, this can be replaced with a database/cloud
# session system for multiple patients.
# ============================================================

medical_session = {

    "active": False,

    "original_file": None,

    "encrypted_file": None,

    "quantum_key": None,

    "qber": None,

    "eve": False,

    "secure": False,

    "medical_type": None,

    "confidence": None

}


patient_session = {
    "active": False,
    "monitoring_started": False,
    "generator": None,
    "patient_id": None,
    "patient_name": None,
    "vitals": None,
    "packet": None,
    "encrypted_file": None,
    "quantum_key": None,
    "qber": None,
    "eve": False,
    "secure": False,
    "bb84_result": None,
    "latest_encrypted_packet": None
}


def encrypt_monitor_packet(packet, quantum_key):
    plaintext = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(encryptor.derive_key(quantum_key)).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_monitor_packet(encrypted_packet, quantum_key):
    encrypted = base64.b64decode(encrypted_packet)
    return json.loads(AESGCM(encryptor.derive_key(quantum_key)).decrypt(encrypted[:12], encrypted[12:], None).decode("utf-8"))


hospital_session = {
    "stage": 1,
    "hospital_a": "Hospital A",
    "hospital_b": "Hospital B",
    "patient_record": None,
    "bb84": None,
    "quantum_key": None,
    "qber": None,
    "eve": False,
    "secure": False,
    "encrypted_payload": None,
    "encrypted_file": None,
    "encrypted": False
}


cloud_session = {
    "stage": 1,
    "record": None,
    "bb84": None,
    "quantum_key": None,
    "qber": None,
    "eve": False,
    "secure": False,
    "encrypted_file": None
}


emergency_session = {
    "stage": 1,
    "record": None,
    "bb84": None,
    "quantum_key": None,
    "qber": None,
    "eve": False,
    "secure": False,
    "encrypted_file": None,
    "ambulance_monitoring": False,
    "ambulance_generator": None,
    "live_vitals": None,
    "latest_encrypted_packet": None,
    "packets_sent": 0,
    "last_packet_timestamp": None,
    "secure_live": False
}


def encrypt_emergency_live_packet(packet, quantum_key):
    plaintext = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(encryptor.derive_key(quantum_key)).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_emergency_live_packet(encrypted_packet, quantum_key):
    encrypted = base64.b64decode(encrypted_packet)
    return json.loads(AESGCM(encryptor.derive_key(quantum_key)).decrypt(encrypted[:12], encrypted[12:], None).decode("utf-8"))


def encrypt_emergency_record(record, quantum_key):
    plaintext = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(encryptor.derive_key(quantum_key)).encrypt(nonce, plaintext, None)
    filename = str(uuid.uuid4()) + ".emergency.enc"
    path = os.path.join(UPLOAD_FOLDER, filename)
    with open(path, "wb") as encrypted_file:
        encrypted_file.write(nonce + ciphertext)
    return path


def create_corrupted_image(input_path, qber=100, fully_corrupt=False):

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    corruption = max(0.0, min(float(qber) / 100, 1.0))
    complete_corruption = fully_corrupt or corruption >= 1.0
    randomizer = random.Random(f"{input_path}:{qber}:{fully_corrupt}")

    if complete_corruption:
        image = Image.effect_noise((width, height), 90).convert("RGB")

    draw = ImageDraw.Draw(image)

    block_width = max(12, width // 8)
    block_height = max(12, height // 8)
    noise_blocks = max(1, int(64 * max(corruption, 0.08)))
    if complete_corruption:
        noise_blocks = 64

    for _ in range(noise_blocks):
        block_x = randomizer.randrange(0, width, block_width)
        block_y = randomizer.randrange(0, height, block_height)
        noise_width = min(block_width, width - block_x)
        noise_height = min(block_height, height - block_y)
        noise = Image.effect_noise((noise_width, noise_height), 100).convert("RGB")
        image.paste(noise, (block_x, block_y))

    dot_area = max(1, (width * height) // 1200)
    dot_count = max(20, int(dot_area * max(corruption, 0.08)))
    if complete_corruption:
        dot_count = max(dot_count, (width * height) // 300)

    for _ in range(dot_count):
        cluster_x = randomizer.randint(0, max(0, width - 1))
        cluster_y = randomizer.randint(0, max(0, height - 1))
        cluster_size = randomizer.randint(2, 6)

        for _ in range(cluster_size):
            center_x = cluster_x + randomizer.randint(-12, 12)
            center_y = cluster_y + randomizer.randint(-12, 12)
            radius = randomizer.randint(1, max(2, min(width, height) // 90))
            dot_color = (0, 0, 0) if randomizer.random() < 0.5 else (255, 255, 255)
            draw.ellipse(
                (center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius),
                fill=dot_color
            )

    output_filename = str(uuid.uuid4()) + "_corrupted.png"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    image.save(output_path, format="PNG")
    return output_filename


# ============================================================
# HEALTH CHECK FOR RENDER
# ============================================================

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"}), 200


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():


    return render_template(
        "index.html"
    )


# ============================================================
# MEDICAL IMAGING PAGE
# ============================================================

@app.route("/medical-imaging")
def medical_imaging():

    return render_template(
        "medical_imaging.html"
    )


@app.route("/patient-monitoring")
def patient_monitoring():

    return render_template(
        "patient_monitoring.html"
    )


@app.route("/emergency-care")
def emergency_care():
    return render_template("emergency_care.html")


@app.route("/api/emergency-care/prepare", methods=["POST"])
def prepare_emergency_care():
    data = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
    required = ["patient_name", "patient_id", "age", "emergency_condition", "chief_complaint", "diagnosis", "treatment", "medication", "doctor", "emergency_summary", "priority", "heart_rate", "spo2", "blood_pressure", "temperature", "respiration_rate"]
    record = {field: str(data.get(field, "")).strip() for field in required}
    if record["priority"] not in {"CRITICAL", "HIGH", "MEDIUM"}:
        return jsonify({"success": False, "message": "Select a valid emergency priority."}), 400
    missing = [field.replace("_", " ").title() for field in required if not record[field]]
    if missing:
        return jsonify({"success": False, "message": "Required fields: " + ", ".join(missing)}), 400
    image = request.files.get("medical_image")
    if image and image.filename:
        extension = os.path.splitext(image.filename)[1].lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
            return jsonify({"success": False, "message": "Unsupported medical image format."}), 400
        record["medical_image"] = {"data": base64.b64encode(image.read()).decode("ascii"), "mime_type": image.mimetype or "application/octet-stream"}
    emergency_session.update({"stage": 2, "record": record, "bb84": None, "quantum_key": None, "qber": None, "eve": False, "secure": False, "encrypted_file": None, "latest_encrypted_packet": None, "secure_live": False})
    return jsonify({"success": True, "message": "Emergency patient data prepared for secure transmission.", "stage": 2})


@app.route("/api/emergency-care/ambulance/start", methods=["POST"])
def start_emergency_ambulance_monitoring():
    generator = VitalSignGenerator()
    emergency_session.update({"ambulance_monitoring": True, "ambulance_generator": generator, "live_vitals": generator.snapshot(), "packets_sent": 0, "latest_encrypted_packet": None, "last_packet_timestamp": None})
    return jsonify({"success": True, "monitoring": True, "vitals": emergency_session["live_vitals"], "message": "LIVE — AMBULANCE MONITORING"})


@app.route("/api/emergency-care/ambulance/stop", methods=["POST"])
def stop_emergency_ambulance_monitoring():
    emergency_session["ambulance_monitoring"] = False
    return jsonify({"success": True, "monitoring": False, "vitals": emergency_session["live_vitals"]})


@app.route("/api/emergency-care/ambulance/vitals", methods=["GET"])
def emergency_ambulance_vitals():
    if not emergency_session["ambulance_monitoring"] or not emergency_session["ambulance_generator"]:
        return jsonify({"success": False, "message": "Ambulance monitoring has not started."}), 409
    emergency_session["live_vitals"] = emergency_session["ambulance_generator"].next()
    if emergency_session["secure_live"] and emergency_session["quantum_key"]:
        packet = {"patient_id": emergency_session["record"]["patient_id"], "patient_name": emergency_session["record"]["patient_name"], "emergency_condition": emergency_session["record"]["emergency_condition"], "priority": emergency_session["record"]["priority"], "timestamp": time.time(), **emergency_session["live_vitals"]}
        emergency_session["latest_encrypted_packet"] = encrypt_emergency_live_packet(packet, emergency_session["quantum_key"])
        emergency_session["packets_sent"] += 1
        emergency_session["last_packet_timestamp"] = packet["timestamp"]
    return jsonify({"success": True, "vitals": emergency_session["live_vitals"], "monitoring": True, "packets_sent": emergency_session["packets_sent"], "last_packet_timestamp": emergency_session["last_packet_timestamp"]})


@app.route("/api/emergency-care/live-status", methods=["GET"])
def emergency_live_status():
    return jsonify({"success": True, "monitoring": emergency_session["ambulance_monitoring"], "secure_live": emergency_session["secure_live"], "packets_sent": emergency_session["packets_sent"], "last_packet_timestamp": emergency_session["last_packet_timestamp"]})


@app.route("/api/emergency-care/bb84", methods=["POST"])
def emergency_care_bb84():
    if not emergency_session["record"]:
        return jsonify({"success": False, "message": "Prepare emergency data first."}), 409
    eve = bool((request.get_json(silent=True) or {}).get("eve", False))
    result = BB84(num_bits=128, eve=eve, noise_rate=0.0).run()
    matching = ["MATCH" if a == b else "DIFFERENT" for a, b in zip(result["alice_bases"], result["bob_bases"])]
    compared = sum(value == "MATCH" for value in matching)
    incorrect = sum(result["alice_bits"][i] != result["bob_bits"][i] for i, value in enumerate(matching) if value == "MATCH")
    details = {"alice_bits": result["alice_bits"], "alice_bases": result["alice_bases"], "bob_bases": result["bob_bases"], "bob_bits": result["bob_bits"], "eve_bases": result.get("eve_bases", []), "eve_bits": result.get("eve_bits", []), "sifted_key": result["sifted_key"], "matching_positions": matching, "compared_bits": compared, "incorrect_bits": incorrect, "total_bits": result["num_bits"]}
    emergency_session.update({"stage": 2, "bb84": details, "quantum_key": result["key"], "qber": result["qber"], "eve": eve, "secure": result["secure"], "encrypted_file": None})
    return jsonify({"success": True, "bb84": details, "quantum_key": result["key"], "key_length": len(result["key"]), "qber": result["qber"], "secure": result["secure"], "eve": eve, "threshold": 11.0, "channel_status": "SECURE" if result["secure"] else "INSECURE"})


@app.route("/api/emergency-care/encrypt", methods=["POST"])
def encrypt_emergency_care():
    if not emergency_session["record"] or not emergency_session["quantum_key"]:
        return jsonify({"success": False, "message": "Generate the BB84 key first."}), 409
    old_file = emergency_session.get("encrypted_file")
    if old_file and os.path.exists(old_file):
        os.remove(old_file)
    emergency_session["encrypted_file"] = encrypt_emergency_record(emergency_session["record"], emergency_session["quantum_key"])
    emergency_session["stage"] = 3
    emergency_session["secure_live"] = emergency_session["secure"] and emergency_session["ambulance_monitoring"]
    if emergency_session["ambulance_monitoring"] and emergency_session["secure_live"]:
        packet = {"patient_id": emergency_session["record"]["patient_id"], "patient_name": emergency_session["record"]["patient_name"], "emergency_condition": emergency_session["record"]["emergency_condition"], "priority": emergency_session["record"]["priority"], "timestamp": time.time(), **emergency_session["live_vitals"]}
        emergency_session["latest_encrypted_packet"] = encrypt_emergency_live_packet(packet, emergency_session["quantum_key"])
        emergency_session["packets_sent"] = 1
        emergency_session["last_packet_timestamp"] = packet["timestamp"]
    return jsonify({"success": True, "encrypted": True, "compromised": not emergency_session["secure"], "message": "Emergency data encrypted." if emergency_session["secure"] else "Emergency data encrypted for demonstration; channel integrity is compromised."})


@app.route("/api/emergency-care/decrypt", methods=["POST"])
def decrypt_emergency_care():
    entered_key = str((request.get_json(silent=True) or {}).get("quantum_key", "")).strip()
    original_key = emergency_session["quantum_key"]
    if not original_key or not emergency_session["encrypted_file"]:
        return jsonify({"success": False, "message": "No encrypted emergency session is available."}), 409
    if not entered_key or not hmac.compare_digest(entered_key, original_key):
        return jsonify({"success": False, "message": "Invalid quantum key. Emergency patient data cannot be decrypted."}), 401
    if emergency_session["secure_live"] and emergency_session["latest_encrypted_packet"]:
        try:
            packet = decrypt_emergency_live_packet(emergency_session["latest_encrypted_packet"], entered_key)
        except Exception:
            return jsonify({"success": False, "message": "Quantum key is incorrect or encrypted emergency data has been modified."}), 401
        receiver_record = dict(emergency_session["record"])
        receiver_record.update(packet)
        return jsonify({"success": True, "decrypted": True, "live": True, "packet": packet, "record": receiver_record, "message": "LIVE EMERGENCY DATA RECEIVED. Secure transmission from ambulance."})

    if not emergency_session["secure"]:
        corrupted = dict(emergency_session["record"])
        corrupted.update({"diagnosis": "[CORRUPTED DATA]", "treatment": "[DATA INTEGRITY FAILURE]", "emergency_summary": "CORRUPTED / UNTRUSTED DATA"})
        corrupted_image = None
        original_image = emergency_session["record"].get("medical_image")
        if original_image:
            try:
                image = Image.open(io.BytesIO(base64.b64decode(original_image["data"]))).convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, max(1, image.width // 2), max(1, image.height // 2)), fill=(180, 20, 35))
                draw.line((0, image.height - 1, image.width - 1, 0), fill=(0, 0, 0), width=max(1, min(image.size) // 12))
                output = io.BytesIO()
                image.save(output, format="PNG")
                corrupted_image = {"data": base64.b64encode(output.getvalue()).decode("ascii"), "mime_type": "image/png"}
            except Exception:
                corrupted_image = None
        corrupted.pop("medical_image", None)
        return jsonify({"success": True, "decrypted": False, "compromised": True, "qber": emergency_session["qber"], "record": corrupted, "medical_image": corrupted_image, "message": "DATA INTEGRITY COMPROMISED. CORRUPTED EMERGENCY DATA. INSECURE CHANNEL — EAVESDROPPING DETECTED."}), 409
    try:
        with open(emergency_session["encrypted_file"], "rb") as encrypted_file:
            encrypted = encrypted_file.read()
        record = json.loads(AESGCM(encryptor.derive_key(entered_key)).decrypt(encrypted[:12], encrypted[12:], None).decode("utf-8"))
    except Exception:
        return jsonify({"success": False, "message": "Emergency patient data cannot be decrypted."}), 401
    image = record.pop("medical_image", None)
    return jsonify({"success": True, "decrypted": True, "record": record, "medical_image": image, "message": "Emergency data decrypted successfully. Secure emergency record received from Hospital A."})


@app.route("/api/emergency-care/reset", methods=["POST"])
def reset_emergency_care():
    encrypted_file = emergency_session.get("encrypted_file")
    if encrypted_file and os.path.exists(encrypted_file):
        os.remove(encrypted_file)
    emergency_session.update({"stage": 1, "record": None, "bb84": None, "quantum_key": None, "qber": None, "eve": False, "secure": False, "encrypted_file": None, "ambulance_monitoring": False, "ambulance_generator": None, "live_vitals": None, "latest_encrypted_packet": None, "packets_sent": 0, "last_packet_timestamp": None, "secure_live": False})
    return jsonify({"success": True})


@app.route("/hospital-sharing")
def hospital_sharing():
    return render_template("hospital_sharing.html")


@app.route("/api/hospital-sharing/prepare", methods=["POST"])
def prepare_hospital_sharing():
    data = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
    required_fields = ["patient_id", "patient_name", "medical_record_number", "problem", "diagnosis", "doctor", "department", "treatment", "patient_summary"]
    record = {field: str(data.get(field, "")).strip() for field in required_fields}
    missing = [field.replace("_", " ").title() for field in required_fields if not record[field]]
    if missing:
        return jsonify({"success": False, "message": "Required fields: " + ", ".join(missing)}), 400

    image = request.files.get("medical_image")
    if image and image.filename:
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
        extension = os.path.splitext(image.filename)[1].lower()
        if extension not in allowed_extensions:
            return jsonify({"success": False, "message": "Unsupported medical image format."}), 400
        record["medical_image"] = {
            "data": base64.b64encode(image.read()).decode("ascii"),
            "mime_type": image.mimetype or "application/octet-stream"
        }

    hospital_session.update({
        "stage": 2,
        "hospital_a": str(data.get("hospital_a", "Hospital A")).strip() or "Hospital A",
        "hospital_b": str(data.get("hospital_b", "Hospital B")).strip() or "Hospital B",
        "patient_record": record,
        "bb84": None,
        "quantum_key": None,
        "qber": None,
        "eve": False,
        "secure": False,
        "encrypted_payload": None,
        "encrypted": False
    })
    return jsonify({"success": True, "stage": 2})


@app.route("/api/hospital-sharing/bb84", methods=["POST"])
def hospital_sharing_bb84():
    if not hospital_session["patient_record"]:
        return jsonify({"success": False, "message": "Prepare the patient record first."}), 409

    data = request.get_json(silent=True) or {}
    eve = bool(data.get("eve", False))
    result = BB84(num_bits=128, eve=eve, noise_rate=0.0).run()
    matching_positions = [
        "MATCH" if alice_basis == bob_basis else "DIFFERENT"
        for alice_basis, bob_basis in zip(result["alice_bases"], result["bob_bases"])
    ]
    compared_bits = sum(position == "MATCH" for position in matching_positions)
    incorrect_bits = sum(
        result["alice_bits"][index] != result["bob_bits"][index]
        for index, position in enumerate(matching_positions)
        if position == "MATCH"
    )
    bb84 = {
        "alice_bits": result["alice_bits"],
        "alice_bases": result["alice_bases"],
        "bob_bases": result["bob_bases"],
        "bob_bits": result["bob_bits"],
        "eve_bases": result.get("eve_bases", []),
        "eve_bits": result.get("eve_bits", []),
        "sifted_key": result["sifted_key"],
        "matching_positions": matching_positions,
        "compared_bits": compared_bits,
        "incorrect_bits": incorrect_bits,
        "total_bits": result["num_bits"]
    }
    hospital_session.update({
        "stage": 2, "bb84": bb84, "quantum_key": result["key"], "qber": result["qber"],
        "eve": eve, "secure": result["secure"], "encrypted_payload": None, "encrypted": False
    })
    return jsonify({
        "success": True, "bb84": bb84, "quantum_key": result["key"], "key_length": len(result["key"]),
        "qber": result["qber"], "secure": result["secure"], "eve": eve, "threshold": 11.0,
        "channel_status": "SECURE" if result["secure"] else "COMPROMISED"
    })


@app.route("/api/hospital-sharing/encrypt", methods=["POST"])
def encrypt_hospital_sharing():
    if not hospital_session["bb84"] or not hospital_session["patient_record"]:
        return jsonify({"success": False, "message": "Generate a BB84 key first."}), 409
    payload = dict(hospital_session["patient_record"])
    payload["hospital_a"] = hospital_session["hospital_a"]
    payload["hospital_b"] = hospital_session["hospital_b"]
    plaintext = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(encryptor.derive_key(hospital_session["quantum_key"])).encrypt(nonce, plaintext, None)
    hospital_session["encrypted_payload"] = base64.b64encode(nonce + ciphertext).decode("ascii")
    encrypted_filename = str(uuid.uuid4()) + ".patient.enc"
    encrypted_path = os.path.join(UPLOAD_FOLDER, encrypted_filename)
    with open(encrypted_path, "wb") as encrypted_file:
        encrypted_file.write(nonce + ciphertext)
    hospital_session["encrypted_file"] = encrypted_path
    hospital_session["encrypted"] = True
    hospital_session["stage"] = 3
    if hospital_session["secure"]:
        message = "Patient record encrypted successfully."
    else:
        message = "Patient data encrypted for demonstration, but the transmission is compromised."
    return jsonify({"success": True, "encrypted": True, "compromised": not hospital_session["secure"], "message": message})


@app.route("/api/hospital-sharing/decrypt", methods=["POST"])
def decrypt_hospital_sharing():
    data = request.get_json(silent=True) or {}
    entered_key = str(data.get("quantum_key", "")).strip()
    original_key = hospital_session["quantum_key"]
    if not original_key or not hospital_session["bb84"]:
        return jsonify({"success": False, "message": "No hospital-sharing session is available."}), 409
    if not entered_key or not hmac.compare_digest(entered_key, original_key):
        return jsonify({"success": False, "decrypted": False, "message": "Invalid quantum key. Patient record could not be decrypted."}), 401
    if not hospital_session["secure"]:
        corrupted = dict(hospital_session["patient_record"])
        corrupted.update({"diagnosis": "[CORRUPTED DATA]", "treatment": "[DATA INTEGRITY FAILURE]", "medical_report": "DEMONSTRATION CORRUPTION ONLY"})
        corrupted_image = None
        original_image = hospital_session["patient_record"].get("medical_image")
        if original_image:
            try:
                image = Image.open(io.BytesIO(base64.b64decode(original_image["data"]))).convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, max(1, image.width // 2), max(1, image.height // 2)), fill=(180, 20, 35))
                draw.line((0, image.height - 1, image.width - 1, 0), fill=(0, 0, 0), width=max(1, min(image.size) // 12))
                output = io.BytesIO()
                image.save(output, format="PNG")
                corrupted_image = {"data": base64.b64encode(output.getvalue()).decode("ascii"), "mime_type": "image/png"}
            except Exception:
                corrupted_image = None
        corrupted.pop("medical_image", None)
        return jsonify({"success": True, "decrypted": False, "compromised": True, "qber": hospital_session["qber"], "record": corrupted, "medical_image": corrupted_image, "message": "DECRYPTION COMPLETED WITH COMPROMISED DATA. The result is corrupted and MUST NOT be trusted."}), 409
    if not hospital_session["encrypted_file"] or not os.path.exists(hospital_session["encrypted_file"]):
        return jsonify({"success": False, "message": "The patient record has not been encrypted."}), 409
    try:
        with open(hospital_session["encrypted_file"], "rb") as encrypted_file:
            encrypted = encrypted_file.read()
        record = json.loads(AESGCM(encryptor.derive_key(entered_key)).decrypt(encrypted[:12], encrypted[12:], None).decode("utf-8"))
    except Exception:
        return jsonify({"success": False, "decrypted": False, "message": "Patient record could not be decrypted."}), 401
    hospital_session["stage"] = 4
    image = record.pop("medical_image", None)
    return jsonify({"success": True, "decrypted": True, "secure": True, "record": record, "medical_image": image, "message": "Decryption successful. Secure channel verified."})


@app.route("/api/hospital-sharing/reset", methods=["POST"])
def reset_hospital_sharing():
    encrypted_file = hospital_session.get("encrypted_file")
    if encrypted_file and os.path.exists(encrypted_file):
        os.remove(encrypted_file)
    hospital_session.update({"stage": 1, "patient_record": None, "bb84": None, "quantum_key": None, "qber": None, "eve": False, "secure": False, "encrypted_payload": None, "encrypted_file": None, "encrypted": False})
    return jsonify({"success": True})


@app.route("/healthcare-cloud")
def healthcare_cloud():
    return render_template("healthcare_cloud.html")


def cloud_stats():
    encrypted = bool(cloud_session["encrypted_file"] and os.path.exists(cloud_session["encrypted_file"]))
    return {
        "records_stored": int(encrypted),
        "encrypted_records": int(encrypted),
        "bb84_sessions": int(cloud_session["bb84"] is not None),
        "secure_channels": int(cloud_session["secure"] and cloud_session["bb84"] is not None),
        "threats_detected": int(cloud_session["eve"] and cloud_session["bb84"] is not None and not cloud_session["secure"])
    }


@app.route("/api/healthcare-cloud/prepare", methods=["POST"])
def prepare_healthcare_cloud():
    data = request.form.to_dict()
    required = ["patient_name", "patient_id", "age", "medical_condition", "diagnosis", "treatment", "doctor_notes", "heart_rate", "spo2", "blood_pressure", "temperature", "respiration_rate"]
    record = {field: str(data.get(field, "")).strip() for field in required}
    missing = [field.replace("_", " ").title() for field in required if not record[field]]
    if missing:
        return jsonify({"success": False, "message": "Required fields: " + ", ".join(missing)}), 400
    for upload_name, record_name in (("medical_image", "medical_image"), ("medical_report", "medical_report")):
        upload = request.files.get(upload_name)
        if upload and upload.filename:
            record[record_name] = {"filename": upload.filename, "data": base64.b64encode(upload.read()).decode("ascii"), "mime_type": upload.mimetype or "application/octet-stream"}
    old_file = cloud_session.get("encrypted_file")
    if old_file and os.path.exists(old_file):
        os.remove(old_file)
    cloud_session.update({"stage": 2, "record": record, "bb84": None, "quantum_key": None, "qber": None, "eve": False, "secure": False, "encrypted_file": None})
    return jsonify({"success": True, "stage": 2, "message": "Cloud record prepared for BB84 encryption."})


@app.route("/api/healthcare-cloud/encrypt", methods=["POST"])
def encrypt_healthcare_cloud():
    if not cloud_session["record"]:
        return jsonify({"success": False, "message": "Prepare the cloud record first."}), 409
    eve = bool((request.get_json(silent=True) or {}).get("eve", False))
    result = BB84(num_bits=128, eve=eve, noise_rate=0.0).run()
    matching = ["KEEP" if alice == bob else "DISCARD" for alice, bob in zip(result["alice_bases"], result["bob_bases"])]
    compared = sum(value == "KEEP" for value in matching)
    incorrect = sum(result["alice_bits"][index] != result["bob_bits"][index] for index, value in enumerate(matching) if value == "KEEP")
    details = {"alice_bits": result["alice_bits"], "alice_bases": result["alice_bases"], "bob_bases": result["bob_bases"], "bob_bits": result["bob_bits"], "eve_bases": result.get("eve_bases", []), "eve_bits": result.get("eve_bits", []), "sifted_key": result["sifted_key"], "matching_positions": matching, "compared_bits": compared, "incorrect_bits": incorrect, "total_bits": result["num_bits"]}
    old_file = cloud_session.get("encrypted_file")
    if old_file and os.path.exists(old_file):
        os.remove(old_file)
    plaintext = json.dumps(cloud_session["record"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    encrypted_file = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + ".cloud.enc")
    nonce = os.urandom(12)
    with open(encrypted_file, "wb") as output:
        output.write(nonce + AESGCM(encryptor.derive_key(result["key"])).encrypt(nonce, plaintext, None))
    cloud_session.update({"stage": 2, "bb84": details, "quantum_key": result["key"], "qber": result["qber"], "eve": eve, "secure": result["secure"], "encrypted_file": encrypted_file})
    return jsonify({"success": True, "bb84": details, "quantum_key": result["key"], "key_length": len(result["key"]), "qber": result["qber"], "secure": result["secure"], "eve": eve, "channel_status": "SECURE" if result["secure"] else "INSECURE", "message": "MEDICAL RECORD ENCRYPTED · STORED IN QUANTUM-SECURE CLOUD", "stats": cloud_stats()})


@app.route("/api/healthcare-cloud/access", methods=["POST"])
def access_healthcare_cloud():
    entered_key = str((request.get_json(silent=True) or {}).get("quantum_key", "")).strip()
    if not cloud_session["quantum_key"] or not cloud_session["encrypted_file"]:
        return jsonify({"success": False, "message": "No encrypted cloud record is available."}), 409
    if not entered_key or not hmac.compare_digest(entered_key, cloud_session["quantum_key"]):
        return jsonify({"success": False, "message": "ACCESS DENIED · DECRYPTION FAILED · INVALID QUANTUM KEY"}), 401
    with open(cloud_session["encrypted_file"], "rb") as encrypted_file:
        encrypted = encrypted_file.read()
    try:
        record = json.loads(AESGCM(encryptor.derive_key(entered_key)).decrypt(encrypted[:12], encrypted[12:], None).decode("utf-8"))
    except Exception:
        return jsonify({"success": False, "message": "ACCESS DENIED · DECRYPTION FAILED"}), 401
    if not cloud_session["secure"]:
        compromised = dict(record)
        compromised["diagnosis"] = "[CORRUPTED DATA]"
        compromised["treatment"] = "[DATA INTEGRITY FAILURE]"
        compromised["doctor_notes"] = "CORRUPTED / UNTRUSTED DATA"
        return jsonify({"success": True, "compromised": True, "record": compromised, "message": "DATA INTEGRITY COMPROMISED · QUANTUM CHANNEL INSECURE · CLOUD RECORD CANNOT BE TRUSTED"}), 409
    return jsonify({"success": True, "secure": True, "record": record, "message": "AUTHORIZED ACCESS · CLOUD RECORD DECRYPTED · DATA INTEGRITY VERIFIED"})


@app.route("/api/healthcare-cloud/reset", methods=["POST"])
def reset_healthcare_cloud():
    encrypted_file = cloud_session.get("encrypted_file")
    if encrypted_file and os.path.exists(encrypted_file):
        os.remove(encrypted_file)
    cloud_session.update({"stage": 1, "record": None, "bb84": None, "quantum_key": None, "qber": None, "eve": False, "secure": False, "encrypted_file": None})
    return jsonify({"success": True})


@app.route("/api/patient-monitoring/start", methods=["POST"])
def start_patient_monitoring():

    data = request.get_json(silent=True) or {}
    patient_id = str(data.get("patient_id", "")).strip()
    patient_name = str(data.get("patient_name", "")).strip()

    if not patient_id or not patient_name:
        return jsonify({
            "success": False,
            "message": "Patient ID and patient name are required."
        }), 400

    generator = VitalSignGenerator()
    patient_session.update({
        "active": True,
        "monitoring_started": True,
        "generator": generator,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "vitals": generator.snapshot(),
        "packet": None,
        "encrypted_file": None,
        "quantum_key": None,
        "qber": None,
        "eve": False,
        "secure": False
        ,"bb84_result": None,
        "latest_encrypted_packet": None
    })

    return jsonify({
        "success": True,
        "monitoring": True,
        "vitals": patient_session["vitals"]
    })


@app.route("/api/patient-monitoring/vitals", methods=["GET"])
def patient_monitoring_vitals():

    if not patient_session["monitoring_started"]:
        return jsonify({
            "success": False,
            "message": "Monitoring has not started."
        }), 400

    patient_session["vitals"] = patient_session["generator"].next()
    if patient_session["secure"] and patient_session["quantum_key"]:
        packet = dict(patient_session["vitals"])
        packet.update({
            "patient_id": patient_session["patient_id"],
            "patient_name": patient_session["patient_name"],
            "timestamp": time.time()
        })
        patient_session["packet"] = packet
        patient_session["latest_encrypted_packet"] = encrypt_monitor_packet(packet, patient_session["quantum_key"])
    return jsonify({
        "success": True,
        "vitals": patient_session["vitals"]
    })


@app.route("/api/patient-monitoring/secure", methods=["POST"])
def secure_patient_monitoring():

    if not patient_session["monitoring_started"]:
        return jsonify({
            "success": False,
            "message": "Start monitoring before securing patient data."
        }), 400

    data = request.get_json(silent=True) or {}
    patient_id = str(data.get("patient_id", "")).strip()
    patient_name = str(data.get("patient_name", "")).strip()
    eve = bool(data.get("eve", False))

    if not patient_id or not patient_name:
        return jsonify({
            "success": False,
            "message": "Patient ID and patient name are required."
        }), 400

    patient_session["patient_id"] = patient_id
    patient_session["patient_name"] = patient_name
    patient_session["eve"] = eve
    patient_session["vitals"] = patient_session["generator"].snapshot()
    packet = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "heart_rate": patient_session["vitals"]["heart_rate"],
        "spo2": patient_session["vitals"]["spo2"],
        "systolic": patient_session["vitals"]["systolic"],
        "diastolic": patient_session["vitals"]["diastolic"],
        "blood_pressure": patient_session["vitals"]["blood_pressure"],
        "temperature": patient_session["vitals"]["temperature"],
        "respiration_rate": patient_session["vitals"]["respiration_rate"],
        "timestamp": time.time()
    }

    quantum_result = BB84(num_bits=128, eve=eve).run()
    quantum_key = quantum_result["key"]
    qber = quantum_result["qber"]
    secure = quantum_result["secure"]
    matching_positions = [
        "✓" if alice_basis == bob_basis else "✗"
        for alice_basis, bob_basis in zip(
            quantum_result["alice_bases"],
            quantum_result["bob_bases"]
        )
    ]
    compared_bits = sum(
        position == "✓"
        for position in matching_positions
    )
    incorrect_bits = sum(
        quantum_result["alice_bits"][index] != quantum_result["bob_bits"][index]
        for index, position in enumerate(matching_positions)
        if position == "✓"
    )
    patient_session.update({
        "packet": packet,
        "quantum_key": quantum_key,
        "qber": qber,
        "secure": secure,
        "active": secure
        ,"latest_encrypted_packet": encrypt_monitor_packet(packet, quantum_key) if secure else None,
        "bb84_result": {
            "alice_bits": quantum_result["alice_bits"],
            "alice_bases": quantum_result["alice_bases"],
            "bob_bases": quantum_result["bob_bases"],
            "bob_bits": quantum_result["bob_bits"],
            "eve_bases": quantum_result.get("eve_bases", []),
            "eve_bits": quantum_result.get("eve_bits", []),
            "sifted_key": quantum_result["sifted_key"],
            "matching_positions": matching_positions,
            "compared_bits": compared_bits,
            "incorrect_bits": incorrect_bits
        }
    })

    encrypted_filename = None
    if secure:
        packet_filename = str(uuid.uuid4()) + ".json"
        encrypted_filename = str(uuid.uuid4()) + ".patient.enc"
        packet_path = os.path.join(UPLOAD_FOLDER, packet_filename)
        encrypted_path = os.path.join(UPLOAD_FOLDER, encrypted_filename)
        with open(packet_path, "w", encoding="utf-8") as packet_file:
            json.dump(packet, packet_file, sort_keys=True, separators=(",", ":"))
        encryptor.encrypt(packet_path, encrypted_path, quantum_key)
        os.remove(packet_path)
        patient_session["encrypted_file"] = encrypted_path

    return jsonify({
        "success": True,
        "qber": qber,
        "secure": secure,
        "eve": eve,
        "quantum_key": quantum_key,
        "key_length": len(quantum_key),
        "bb84": patient_session["bb84_result"],
        "channel_status": "SECURE" if secure else "COMPROMISED",
        "encrypted": secure,
        "message": "Patient data encrypted securely." if secure else "Eavesdropping detected. Patient data transmission cannot be trusted."
    })


@app.route("/api/patient-monitoring/decrypt", methods=["POST"])
def decrypt_patient_monitoring():

    data = request.get_json(silent=True) or {}
    entered_key = str(data.get("quantum_key", "")).strip()
    original_key = patient_session["quantum_key"]

    if not entered_key or not original_key:
        return jsonify({
            "success": False,
            "message": "No secure patient-data session is available."
        }), 400

    if not hmac.compare_digest(entered_key, original_key):
        return jsonify({
            "success": False,
            "decrypted": False,
            "message": "Invalid quantum key. Patient data could not be decrypted."
        }), 401

    if patient_session["secure"] and patient_session["latest_encrypted_packet"]:
        try:
            packet = decrypt_monitor_packet(patient_session["latest_encrypted_packet"], entered_key)
        except Exception:
            return jsonify({
                "success": False,
                "decrypted": False,
                "message": "Quantum key is incorrect or encrypted patient data has been modified."
            }), 401
        return jsonify({
            "success": True,
            "decrypted": True,
            "packet": packet,
            "message": "Patient data decrypted successfully."
        })

    if not patient_session["secure"]:
        corrupted_packet = dict(patient_session["packet"])
        corrupted_packet["heart_rate"] = "--"
        corrupted_packet["spo2"] = "--"
        corrupted_packet["blood_pressure"] = "CORRUPTED"
        corrupted_packet["temperature"] = "--"
        corrupted_packet["respiration_rate"] = "--"
        return jsonify({
            "success": True,
            "decrypted": False,
            "compromised": True,
            "qber": patient_session["qber"],
            "packet": corrupted_packet,
            "message": "⚠ DECRYPTION COMPLETED WITH COMPROMISED DATA. Eavesdropping was detected. The resulting patient data is corrupted and must not be trusted."
        }), 409

    if not patient_session["encrypted_file"]:
        return jsonify({
            "success": False,
            "message": "No encrypted patient data is available."
        }), 400

    output_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + ".json")
    decryptor.decrypt(
        patient_session["encrypted_file"],
        output_path,
        entered_key
    )
    with open(output_path, "r", encoding="utf-8") as packet_file:
        packet = json.load(packet_file)
    os.remove(output_path)

    return jsonify({
        "success": True,
        "decrypted": True,
        "packet": packet,
        "message": "Patient data decrypted successfully."
    })


# ============================================================
# STEP 1
# MEDICAL IMAGE ANALYSIS
# ============================================================

@app.route(
    "/api/detect-medical-image",
    methods=["POST"]
)
def detect_medical_image():

    try:

        medical_session.update({
            "active": False,
            "original_file": None,
            "encrypted_file": None,
            "quantum_key": None,
            "qber": None,
            "eve": False,
            "secure": False,
            "medical_type": None,
            "confidence": None
        })

        if "medical_image" not in request.files:

            return jsonify({

                "success": False,

                "is_medical": False,

                "message":
                    "No image was uploaded."

            }), 400


        file = request.files[
            "medical_image"
        ]


        if file.filename == "":

            return jsonify({

                "success": False,

                "is_medical": False,

                "message":
                    "No image was selected."

            }), 400


        # ----------------------------------------------------
        # ALLOWED FORMATS (IMAGES & PDF REPORTS)
        # ----------------------------------------------------

        allowed_extensions = {

            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".gif",
            ".tif",
            ".tiff",
            ".pdf"

        }


        extension = os.path.splitext(
            file.filename
        )[1].lower()


        if extension not in allowed_extensions:

            return jsonify({

                "success": False,

                "is_medical": False,

                "message":
                    "Unsupported file format. Please upload JPG, PNG, WEBP, or PDF medical reports."

            }), 400


        # ----------------------------------------------------
        # SAVE TEMPORARY FILE
        # ----------------------------------------------------

        filename = (
            str(uuid.uuid4())
            + extension
        )


        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        file.save(filepath)


        # ----------------------------------------------------
        # MEDICAL CONTENT VALIDATION LAYER
        # ----------------------------------------------------

        validation = validator.validate_file(
            filepath,
            original_filename=file.filename
        )



        # ----------------------------------------------------
        # NON-MEDICAL FILE / REPORT REJECTED
        # ----------------------------------------------------

        if not validation["is_medical"]:
            medical_session["active"] = False
            # Delete rejected file
            try:
                os.remove(filepath)
            except Exception:
                pass

            return jsonify({
                "success": True,
                "is_medical": False,
                "medical_verified": False,
                "content_type": "non_medical",
                "organ": None,
                "modality": None,
                "type": validation.get("medical_type", "Non-Medical Image"),
                "confidence": 0.0,
                "encryption_allowed": False,
                "message": validation.get("message", "This image does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report.")
            })



        # Reset session state on new file upload to prevent cross-request contamination
        medical_session.clear()

        # ----------------------------------------------------
        # MEDICAL FILE ACCEPTED
        # ----------------------------------------------------

        medical_session["original_file"] = filepath

        medical_session["stage1_info"] = validation

        medical_session["medical_type"] = (
            validation["medical_type"]
        )

        medical_session["confidence"] = (
            validation["confidence"]
        )

        medical_session["is_pdf"] = (
            validation.get("is_pdf", False)
        )

        medical_session["active"] = True


        return jsonify({
            "success": True,
            "is_medical": True,
            "medical_verified": True,
            "is_pdf": validation.get("is_pdf", False),
            "content_type": validation.get("input_type", "medical_image"),
            "organ": validation.get("body_region", "Dynamic Medical Imaging"),
            "modality": validation.get("modality", "Medical Diagnostic Image"),
            "type": validation["medical_type"],
            "input_type": validation.get("input_type", "medical_image"),
            "body_region": validation.get("body_region", "Dynamic Medical Imaging"),
            "report_type": validation.get("report_type"),
            "certainty": validation.get("certainty", "high"),
            "confidence": "HIGH" if validation.get("certainty") == "high" else "MEDIUM",
            "encryption_allowed": True,
            "message": validation.get("message", "Medical Content Verified. Ready for quantum encryption.")
        })






    except Exception as error:

        print(
            "\nMEDICAL IMAGE ANALYSIS ERROR:"
        )

        print(error)


        return jsonify({

            "success": False,

            "is_medical": False,

            "message":
                "Unable to analyze the uploaded image.",

            "error":
                str(error)

        }), 500


# ============================================================
# STEP 2
# ENCRYPT MEDICAL IMAGE
# ============================================================

@app.route(
    "/api/encrypt-medical-image",
    methods=["POST"]
)
def encrypt_medical_image():

    try:

        # ----------------------------------------------------
        # CHECK ANALYZED IMAGE
        # ----------------------------------------------------

        if not medical_session["active"]:

            return jsonify({

                "success": False,

                "message":
                    "No analyzed medical image is available."

            }), 400


        image_path = (
            medical_session["original_file"]
        )


        if not image_path or not os.path.exists(
            image_path
        ):

            return jsonify({

                "success": False,

                "message":
                    "Medical image could not be found."

            }), 400


        # ----------------------------------------------------
        # EVE SETTING
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}


        eve = bool(
            data.get(
                "eve",
                False
            )
        )


        medical_session["eve"] = eve


        print(
            "\n================================"
        )

        print(
            "MEDICAL IMAGE ENCRYPTION"
        )

        print(
            "================================"
        )

        print(
            "Eve:",
            eve
        )


        # ----------------------------------------------------
        # BB84
        # ----------------------------------------------------

        bb84 = BB84(

            num_bits=128,

            eve=eve,

            noise_rate=0.0

        )


        result = bb84.run()


        quantum_key = result["key"]

        qber = result["qber"]

        secure = result["secure"]

        matching_bases = sum(
            alice_basis == bob_basis
            for alice_basis, bob_basis in zip(
                result["alice_bases"],
                result["bob_bases"]
            )
        )


        print(
            "Key Length:",
            len(quantum_key)
        )

        print(
            "QBER:",
            qber,
            "%"
        )

        print(
            "Secure:",
            secure
        )


        # ----------------------------------------------------
        # STORE BB84 SESSION INFORMATION
        # ----------------------------------------------------

        medical_session["quantum_key"] = (
            quantum_key
        )

        medical_session["qber"] = qber

        medical_session["secure"] = secure


        # ----------------------------------------------------
        # CREATE ENCRYPTED FILE
        # ----------------------------------------------------

        encrypted_filename = (
            str(uuid.uuid4())
            + ".enc"
        )


        encrypted_path = os.path.join(
            UPLOAD_FOLDER,
            encrypted_filename
        )


        # ----------------------------------------------------
        # IMPORTANT
        #
        # Even when Eve is active, we create the encrypted
        # demonstration data.
        #
        # The compromised status is handled during
        # decryption/output.
        # ----------------------------------------------------

        encryption_result = encryptor.encrypt(

            input_file=image_path,

            output_file=encrypted_path,

            quantum_key=quantum_key

        )


        medical_session["encrypted_file"] = (
            encrypted_path
        )


        print(
            "Encrypted file:",
            encrypted_path
        )


        # ----------------------------------------------------
        # SECURITY STATUS
        # ----------------------------------------------------

        if secure:

            security_status = "SECURE"

        else:

            security_status = "COMPROMISED"


        return jsonify({

            "success": True,

            "medical_type":
                medical_session["medical_type"],

            "confidence":
                medical_session["confidence"],

            "quantum_key":
                quantum_key,

            "key_length":
                len(quantum_key),

            "alice_bits":
                result["alice_bits"],

            "alice_bases":
                result["alice_bases"],

            "bob_bases":
                result["bob_bases"],

            "bob_bits":
                result["bob_bits"],

            "sifted_key":
                result["sifted_key"],

            "matching_bases":
                matching_bases,

            "discarded_bases":
                result["num_bits"] - matching_bases,

            "qber":
                qber,

            "eve":
                eve,

            "secure":
                secure,

            "security_status":
                security_status,

            "encryption":
                encryption_result["algorithm"],

            "encrypted_file":
                encrypted_filename,

            "message":
                "Medical image encryption completed."

        })


    except Exception as error:

        print(
            "\nMEDICAL IMAGE ENCRYPTION ERROR:"
        )

        print(error)


        return jsonify({

            "success": False,

            "message":
                "Medical image encryption failed.",

            "error":
                str(error)

        }), 500


# ============================================================
# STEP 3
# DECRYPT MEDICAL IMAGE
# ============================================================

@app.route(
    "/api/decrypt-medical-image",
    methods=["POST"]
)
def decrypt_medical_image():

    try:

        # ----------------------------------------------------
        # CHECK ACTIVE SESSION
        # ----------------------------------------------------

        if not medical_session["active"]:

            return jsonify({

                "success": False,

                "message":
                    "No active medical security session."

            }), 400


        encrypted_file = (
            medical_session["encrypted_file"]
        )


        if not encrypted_file:

            return jsonify({

                "success": False,

                "message":
                    "No encrypted medical image is available."

            }), 400


        # ----------------------------------------------------
        # PATIENT ENTERS ONLY QUANTUM KEY
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}


        entered_key = str(
            data.get(
                "quantum_key",
                ""
            )
        ).strip()


        if not entered_key:

            return jsonify({

                "success": False,

                "message":
                    "Please enter the quantum key."

            }), 400


        # ----------------------------------------------------
        # GET SESSION INFORMATION
        # ----------------------------------------------------

        original_key = (
            medical_session["quantum_key"]
        )

        eve = (
            medical_session["eve"]
        )

        qber = (
            medical_session["qber"]
        )

        secure = (
            medical_session["secure"]
        )


        # ====================================================
        # CASE 1
        # WRONG QUANTUM KEY
        # ====================================================

        if entered_key != original_key:

            print(
                "\nWRONG QUANTUM KEY"
            )


            return jsonify({

                "success": True,

                "decrypted": False,

                "corrupted": True,

                "reason":
                    "KEY_MISMATCH",

                "qber":
                    qber,

                "eve":
                    eve,

                "security_status":
                    "KEY_MISMATCH",

                "image_url":
                    "/uploads/" + create_corrupted_image(
                        medical_session["original_file"],
                        qber
                    ),

                "message":
                    "Quantum key does not match. "
                    "Original medical image cannot be recovered."

            })


        # ====================================================
        # CASE 2
        # EVE / COMPROMISED CHANNEL
        # ====================================================

        if eve or not secure:

            print(
                "\nCOMPROMISED BB84 SESSION"
            )


            return jsonify({

                "success": True,

                "decrypted": False,

                "corrupted": True,

                "reason":
                    "EAVESDROPPING_DETECTED",

                "qber":
                    qber,

                "eve":
                    eve,

                "security_status":
                    "COMPROMISED",

                "image_url":
                    "/uploads/" + create_corrupted_image(
                        medical_session["original_file"],
                        qber
                    ),

                "message":
                    "Eavesdropping or channel compromise "
                    "was detected. Corrupted medical output "
                    "will be displayed."

            })


        # ====================================================
        # CASE 3
        # CORRECT KEY + SECURE CHANNEL
        # ====================================================

        output_filename = (
            str(uuid.uuid4())
            + "_decrypted"
            + os.path.splitext(
                medical_session["original_file"]
            )[1]
        )


        output_path = os.path.join(
            UPLOAD_FOLDER,
            output_filename
        )


        decryption_result = decryptor.decrypt(

            encrypted_file=encrypted_file,

            output_file=output_path,

            quantum_key=entered_key

        )


        print(
            "\nMEDICAL IMAGE / REPORT DECRYPTED SUCCESSFULLY"
        )


        # ----------------------------------------------------
        # PDF PREVIEW VS IMAGE PREVIEW PREPARATION
        # ----------------------------------------------------

        is_pdf = output_path.lower().endswith(".pdf")
        pdf_url = None

        if is_pdf:
            pdf_url = "/uploads/" + output_filename
            preview_filename = str(uuid.uuid4()) + "_preview.png"
            preview_path = os.path.join(UPLOAD_FOLDER, preview_filename)
            try:
                validator.report_processor.pdf_to_preview_image(output_path, preview_path)
                image_url = "/uploads/" + preview_filename
            except Exception as err:
                print(f"[APP] PDF preview render error: {err}")
                image_url = "/uploads/" + output_filename
        else:
            image_url = "/uploads/" + output_filename


        # ----------------------------------------------------
        # MEDICAL AI ANALYSIS USING MEDGEMMA
        # (Executed ONLY after successful decryption)
        # ----------------------------------------------------

        stage1_info = medical_session.get("stage1_info") or {
            "is_medical": True,
            "input_type": "medical_image",
            "body_region": "Unknown / Unable to determine",
            "modality": medical_session.get("medical_type", "Medical File"),
            "report_type": None,
            "certainty": "medium"
        }

        ai_analysis = medgemma_analyzer.analyze_medical_data(
            output_path,
            stage1_info=stage1_info,
            original_file_path=medical_session.get("original_file")
        )




        return jsonify({

            "success": True,

            "decrypted": True,

            "corrupted": False,

            "reason":
                "SUCCESS",

            "qber":
                qber,

            "eve":
                eve,

            "security_status":
                "SECURE",

            "image_url":
                image_url,

            "pdf_url":
                pdf_url,

            "is_pdf":
                is_pdf,

            "ai_analysis":
                ai_analysis,

            "message":
                "Medical image decrypted successfully. MedGemma AI analysis completed."

        })





    except Exception as error:

        print(
            "\nMEDICAL IMAGE DECRYPTION ERROR:"
        )

        print(error)


        return jsonify({

            "success": False,

            "decrypted": False,

            "corrupted": True,

            "message":
                "Medical image decryption failed.",

            "error":
                str(error)

        }), 500


# ============================================================
# SERVE GENERATED MEDICAL IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    from flask import send_from_directory

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )