'''
JL3562@cornell.edu
generates some pairs of data based on system time... for testing
'''
import numpy as np
import time

class MockDataGenerator:
    def __init__(self):
        curTime= int(time.time())
        print(f"Using system time in ns as seed: {curTime}")
        self.rng= np.random.RandomState(curTime)

    def generateNoisedLinearPair(self, x0, sigma=3):
        return np.asarray([x0, self.rng.normal(x0, sigma)])

if __name__ == "__main__":
    NSamples = 50
    print(f"generating {NSamples} random numbers:")
    mdg= MockDataGenerator()
    for i in range(NSamples):
        print(mdg.generateNoisedLinearPair(i))