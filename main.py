from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model("model.h5")

# Load the tokenizer (you may need to adjust this based on how you saved it during training)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)  # texts: List of all training texts

# Function for text preprocessing


def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence


# Initialize the FastAPI app
app = FastAPI()

# Create an endpoint for predictions


@app.post("/predict")
def predict(text: str):
    try:
        preprocessed_text = preprocess_text(text)
        prediction = model.predict(preprocessed_text)
        predicted_class = label_encoder.classes_[prediction.argmax()]
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}
