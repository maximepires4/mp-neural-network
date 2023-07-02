import random


def shuffle(input, output):
    combined = list(zip(input, output))
    random.shuffle(combined)
    input[:], output[:] = zip(*combined)
    return input, output
