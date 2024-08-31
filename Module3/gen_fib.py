import numpy as np

# Generate Fibonacci sequences using numpy
def generate_fibonacci_sequence(length):
    fib_sequence = np.zeros(length, dtype=int)
    fib_sequence[0], fib_sequence[1] = 0, 1
    for i in range(2, length):
        fib_sequence[i] = fib_sequence[i-1] + fib_sequence[i-2]
    return fib_sequence

# Test Fibonacci sequence generation
def test_fibonacci_sequence():
    # Generate a sequence of length 10
    length = 10
    fib_sequence = generate_fibonacci_sequence(length)
    expected_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]  # Expected result for the first 10 Fibonacci numbers
    assert np.array_equal(fib_sequence, expected_sequence), "Fibonacci sequence generation failed!"
    print("Fibonacci sequence generation is correct.")

test_fibonacci_sequence()
