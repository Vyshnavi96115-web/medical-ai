import random


class VitalSignGenerator:

    def __init__(self):
        self.values = {
            "heart_rate": 76,
            "spo2": 98,
            "systolic": 119,
            "diastolic": 76,
            "temperature": 36.8,
            "respiration_rate": 16
        }
        self.random = random.Random()

    def next(self):
        self.values["heart_rate"] = self._step(
            self.values["heart_rate"], 55, 110, 2
        )
        self.values["spo2"] = self._step(
            self.values["spo2"], 94, 100, 1
        )
        self.values["systolic"] = self._step(
            self.values["systolic"], 90, 140, 2
        )
        self.values["diastolic"] = self._step(
            self.values["diastolic"], 55, 90, 2
        )
        self.values["temperature"] = round(
            self._step(self.values["temperature"], 36.0, 38.0, 0.1),
            1
        )
        self.values["respiration_rate"] = self._step(
            self.values["respiration_rate"], 10, 24, 1
        )

        return self.snapshot()

    def snapshot(self):
        values = dict(self.values)
        values["blood_pressure"] = (
            f"{values['systolic']}/{values['diastolic']}"
        )
        return values

    def _step(self, value, minimum, maximum, step):
        change = self.random.choice([-step, 0, step])
        return max(minimum, min(maximum, value + change))
