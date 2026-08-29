import random


class BB84:

    def __init__(self, num_bits=128, eve=False, noise_rate=0.0):

        self.num_bits = num_bits
        self.eve = eve
        self.noise_rate = noise_rate

        self.alice_bits = []
        self.alice_bases = []

        self.eve_bases = []
        self.eve_bits = []

        self.bob_bases = []
        self.bob_bits = []

        self.sifted_key = []

        self.qber = 0.0
        self.secure = False

    # ==========================================================
    # ALICE
    # ==========================================================

    def generate_alice(self):

        self.alice_bits = [
            random.randint(0, 1)
            for _ in range(self.num_bits)
        ]

        self.alice_bases = [
            random.randint(0, 1)
            for _ in range(self.num_bits)
        ]

    # ==========================================================
    # BOB
    # ==========================================================

    def generate_bob(self):

        self.bob_bases = [
            random.randint(0, 1)
            for _ in range(self.num_bits)
        ]

    # ==========================================================
    # EVE
    # ==========================================================

    def eve_intercept(self):

        self.eve_bases = [
            random.randint(0, 1)
            for _ in range(self.num_bits)
        ]

        self.eve_bits = []

        for i in range(self.num_bits):

            if self.eve_bases[i] == self.alice_bases[i]:

                measured_bit = self.alice_bits[i]

            else:

                measured_bit = random.randint(0, 1)

            self.eve_bits.append(measured_bit)

    # ==========================================================
    # BOB MEASUREMENT
    # ==========================================================

    def measure(self):

        self.bob_bits = []

        for i in range(self.num_bits):

            # ------------------------------------------
            # Eve present
            # ------------------------------------------

            if self.eve:

                transmitted_bit = self.eve_bits[i]

                transmitted_basis = self.eve_bases[i]

            else:

                transmitted_bit = self.alice_bits[i]

                transmitted_basis = self.alice_bases[i]

            # ------------------------------------------
            # Bob measures
            # ------------------------------------------

            if self.bob_bases[i] == transmitted_basis:

                measured_bit = transmitted_bit

            else:

                measured_bit = random.randint(0, 1)

            # ------------------------------------------
            # Channel noise
            # ------------------------------------------

            if random.random() < self.noise_rate:

                measured_bit = 1 - measured_bit

            self.bob_bits.append(measured_bit)

    # ==========================================================
    # KEY SIFTING
    # ==========================================================

    def sift_key(self):

        self.sifted_key = []

        for i in range(self.num_bits):

            if self.alice_bases[i] == self.bob_bases[i]:

                self.sifted_key.append(
                    self.alice_bits[i]
                )

    # ==========================================================
    # QBER
    # ==========================================================

    def calculate_qber(self):

        errors = 0
        compared_bits = 0

        for i in range(self.num_bits):

            if self.alice_bases[i] == self.bob_bases[i]:

                compared_bits += 1

                if self.alice_bits[i] != self.bob_bits[i]:

                    errors += 1

        if compared_bits == 0:

            self.qber = 0.0

        else:

            self.qber = (
                errors / compared_bits
            ) * 100

        return round(self.qber, 2)

    # ==========================================================
    # SECURITY DECISION
    # ==========================================================

    def security_check(self):

        # BB84 security threshold used by our simulation
        threshold = 11.0

        if self.qber < threshold:

            self.secure = True

        else:

            self.secure = False

        return self.secure

    # ==========================================================
    # GENERATE FINAL KEY
    # ==========================================================

    def get_key(self):

        return ''.join(
            str(bit)
            for bit in self.sifted_key
        )

    # ==========================================================
    # RUN COMPLETE BB84 PROTOCOL
    # ==========================================================

    def run(self):

        # Step 1
        self.generate_alice()

        # Step 2
        self.generate_bob()

        # Step 3
        if self.eve:

            self.eve_intercept()

        # Step 4
        self.measure()

        # Step 5
        self.sift_key()

        # Step 6
        self.calculate_qber()

        # Step 7
        self.security_check()

        return {
            "num_bits": self.num_bits,

            "alice_bits": self.alice_bits,

            "alice_bases": self.alice_bases,

            "bob_bases": self.bob_bases,

            "bob_bits": self.bob_bits,

            "eve_bases": self.eve_bases,

            "eve_bits": self.eve_bits,

            "sifted_key": self.sifted_key,

            "key": self.get_key(),

            "qber": self.qber,

            "secure": self.secure,

            "eve": self.eve
        }


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("BB84 QUANTUM KEY DISTRIBUTION")
    print("==============================\n")

    print("WITHOUT EVE")

    bb84 = BB84(
        num_bits=128,
        eve=False
    )

    result = bb84.run()

    print("Key Length:", len(result["key"]))
    print("QBER:", result["qber"], "%")
    print("Secure:", result["secure"])


    print("\n------------------------------")

    print("WITH EVE")

    bb84_eve = BB84(
        num_bits=128,
        eve=True
    )

    result_eve = bb84_eve.run()

    print("Key Length:", len(result_eve["key"]))
    print("QBER:", result_eve["qber"], "%")
    print("Secure:", result_eve["secure"])