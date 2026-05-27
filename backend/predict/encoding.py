import numpy as np

# One-Hot Encoding Function
def one_hot_encode(sequence):

    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }

    sequence = sequence.upper()
    
    encoded = np.array([
        mapping[nucleotide]
        for nucleotide in sequence
    ])

    return encoded