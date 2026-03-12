
import math
import random


class Circle:

    def __init__(self):
        self.start = 0
        self.end = 2 * math.pi

        self.length = self.end - self.start
        pass

    def Random_Sample(self):

        return self.start + random.random() * self.length


def Segment:
    def __init__(self, start=0, end=1):
        self.start = start
        self.end = end
        self.length = end - start

    def Random_Sample(self):
        return self.start + random.random() * self.length


def Decart:
    def __init(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def Random_Sample(self):
        return [lhs.Random_Sample(), rhs.Random_Sample()]


class Manifold:

    def __init__(self, handels=1):
        pass
