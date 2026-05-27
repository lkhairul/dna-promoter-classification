import numpy as np
from tensorflow.keras.models import load_model
from encoding import one_hot_encode

# Load CNN Model
model = load_model("../model/promoter_cnn_model.h5")

# Prediction Function
def predict_promoter(sequence):

    sequence = sequence.upper()

    # Validate Sequence Length
    if len(sequence) != 57:
        raise ValueError(
            "DNA sequence must contain exactly 57 nucleotides."
        )

    # Encode DNA Sequence
    encoded_sequence = one_hot_encode(sequence)
    encoded_sequence = np.expand_dims(
        encoded_sequence,
        axis=0
    )

    # Predict Probability
    prediction = model.predict(
        encoded_sequence,
        verbose=0
    )[0][0]

    # Determine Label
    if prediction > 0.5:
        label = "Promoter"
        confidence = prediction
    else:
        label = "Non-Promoter"
        confidence = 1 - prediction

    # Return Prediction Result
    return {
        "prediction": label,
        "confidence": round(float(confidence), 4)
    }