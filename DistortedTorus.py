import sympy as sp
import random

import math

from RiemannInterpolation import SymbolicRiemanFromMetric


def norm(l, r):
    tmp = 0
    for i in range(3):
        tmp += (l[i]-r[i])**2
    return tmp ** (0.5)


def DistortionID(z):
    return z


class Torus:
    def __init__(self, Distortion=DistortionID):
        self.w = [sp.symbols('u'), sp.symbols('v')]

        self.r = 1
        self.R = 2

        self.z = [sp.cos(self.w[1]) * (self.R + self.r * sp.cos(self.w[0])),
                  sp.sin(self.w[1]) * (self.R + self.r * sp.cos(self.w[0])),
                  self.r * sp.sin(self.w[0])]

        self.z = Distortion(self.z)

        self.g = []
        for i in range(2):
            self.g.append([])
            for j in range(2):
                self.g[-1].append(0)
                for k in range(3):
                    self.g[-1][-1] += sp.diff(self.z[k], self.w[i]) * \
                        sp.diff(self.z[k], self.w[j])
                    self.g[-1][-1] = sp.simplify(self.g[-1][-1])
        self.Rieman = SymbolicRiemanFromMetric(self.g, self.w, self.w)

    def Distance(self, lhsw, rhsw):
        lhs = [self.z[i].subs(self.w[0], lhsw[0]).subs(
            self.w[1], lhsw[1]) for i in range(3)]
        rhs = [self.z[i].subs(self.w[0], rhsw[0]).subs(
            self.w[1], lhsw[1]) for i in range(3)]

        return norm(lhs, rhs)

    def EspilonNet(self, r):
        N = int(10000 / r)
        pc = [[2 * math.pi * random.random(), 2 * math.pi * random.random()]
              for i in range(N)]
        net = []
        while pc:
            curr = pc.pop(0)
            net.append(curr)
            pc = [pt for pt in pc if self.Distance(curr, pt) > r]
        return pc
