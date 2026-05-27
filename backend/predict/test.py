from predict import predict_promoter

# Sample DNA Sequences
sample_sequences = [
    "A" * 57,
    "CGTA" * 14 + "C",
    "TTGCA" * 11 + "TT"
]

for i, sequence in enumerate(sample_sequences, start=1):
    print(f"\nSample {i}")
    print("Sequence Length:", len(sequence))
    
    result = predict_promoter(sequence)

    print("Prediction :", result["prediction"])
    print("Confidence :", result["confidence"])