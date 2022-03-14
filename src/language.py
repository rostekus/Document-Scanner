from tensorflow import keras
import joblib
import numpy as np


class LanguageDetect:
    def __init__(self):
        self.model = keras.models.load_model(
            "/Users/rostyslavmosorov/Desktop/projekty/receipt-scanner/src/data/langNN.h5"
        )
        self.vect = joblib.load(
            "/Users/rostyslavmosorov/Desktop/projekty/receipt-scanner/src/data/vectorizer.joblib"
        )
        self.lang = ["deu", "eng", "fra", "ita", "por", "spa"]

    def predict(self, text: str):
        x = self.vect.transform([text])
        return self.lang[np.argmax(self.model.predict(x))]


if __name__ == "__main__":
    detect = LanguageDetect()
    print(detect.predict("como te llamas?"))
