import os
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class MedicalImageDecryptor:

    def __init__(self):

        self.algorithm = "AES-256-GCM"


    # ==========================================================
    # DERIVE AES KEY FROM BB84 QUANTUM KEY
    # ==========================================================

    def derive_key(self, quantum_key):

        if not quantum_key:

            raise ValueError(
                "Quantum key was not provided."
            )

        quantum_key = quantum_key.strip()

        key = hashlib.sha256(
            quantum_key.encode("utf-8")
        ).digest()

        return key


    # ==========================================================
    # DECRYPT MEDICAL IMAGE
    # ==========================================================

    def decrypt(
        self,
        encrypted_file,
        output_file,
        quantum_key
    ):

        # ------------------------------------------------------
        # CHECK ENCRYPTED FILE
        # ------------------------------------------------------

        if not os.path.exists(encrypted_file):

            raise FileNotFoundError(
                "Encrypted medical image was not found."
            )


        # ------------------------------------------------------
        # READ ENCRYPTED DATA
        # ------------------------------------------------------

        with open(
            encrypted_file,
            "rb"
        ) as file:

            encrypted_data = file.read()


        # ------------------------------------------------------
        # CHECK DATA SIZE
        # ------------------------------------------------------

        if len(encrypted_data) < 13:

            raise ValueError(
                "Invalid encrypted medical image."
            )


        # ------------------------------------------------------
        # EXTRACT NONCE
        # ------------------------------------------------------

        nonce = encrypted_data[:12]

        ciphertext = encrypted_data[12:]


        # ------------------------------------------------------
        # DERIVE AES-256 KEY
        # ------------------------------------------------------

        aes_key = self.derive_key(
            quantum_key
        )


        # ------------------------------------------------------
        # AES-GCM DECRYPTION
        # ------------------------------------------------------

        aes = AESGCM(aes_key)


        try:

            decrypted_data = aes.decrypt(
                nonce,
                ciphertext,
                None
            )

        except Exception:

            # Wrong quantum key or
            # modified encrypted data

            raise ValueError(
                "Quantum key is incorrect or "
                "encrypted medical data has been modified."
            )


        # ------------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # ------------------------------------------------------

        with open(
            output_file,
            "wb"
        ) as file:

            file.write(decrypted_data)


        return {

            "success": True,

            "algorithm":
                self.algorithm,

            "output_file":
                output_file,

            "output_size":
                len(decrypted_data)

        }


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    print(
        "Medical Image AES Decryption Module Ready"
    )