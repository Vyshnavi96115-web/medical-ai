import os
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class MedicalImageEncryptor:

    def __init__(self):

        self.algorithm = "AES-256-GCM"


    def derive_key(self, quantum_key):

        """
        Convert the BB84-generated quantum key
        into a 256-bit AES key.
        """

        if not quantum_key:

            raise ValueError(
                "Quantum key is empty."
            )


        key = hashlib.sha256(
            quantum_key.encode("utf-8")
        ).digest()


        return key


    def encrypt(
        self,
        input_file,
        output_file,
        quantum_key
    ):

        if not os.path.exists(input_file):

            raise FileNotFoundError(
                "Medical image not found."
            )


        # Read original medical image

        with open(
            input_file,
            "rb"
        ) as file:

            image_data = file.read()


        # Convert BB84 key into AES-256 key

        aes_key = self.derive_key(
            quantum_key
        )


        # AES-GCM requires a unique nonce

        nonce = os.urandom(12)


        aes = AESGCM(aes_key)


        encrypted_data = aes.encrypt(
            nonce,
            image_data,
            None
        )


        # Store:
        #
        # nonce + encrypted image
        #

        with open(
            output_file,
            "wb"
        ) as file:

            file.write(nonce)
            file.write(encrypted_data)


        return {

            "success": True,

            "algorithm":
                self.algorithm,

            "input_size":
                len(image_data),

            "encrypted_size":
                len(encrypted_data) + len(nonce)

        }

    def decrypt(
        self,
        input_file,
        output_file,
        quantum_key
    ):

        if not os.path.exists(input_file):

            raise FileNotFoundError(
                "Encrypted medical image not found."
            )

        with open(input_file, "rb") as file:

            encrypted_data = file.read()

        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        if len(nonce) != 12 or not ciphertext:

            raise ValueError(
                "Invalid encrypted medical image."
            )

        aes = AESGCM(
            self.derive_key(quantum_key)
        )
        image_data = aes.decrypt(
            nonce,
            ciphertext,
            None
        )

        with open(output_file, "wb") as file:

            file.write(image_data)

        return {

            "success": True,

            "output_size": len(image_data)

        }


if __name__ == "__main__":

    print(
        "Medical Image AES Encryption Module Ready"
    )