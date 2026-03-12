import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


def X_Y_Z(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return xs, ys, zs


def Scallar(lhs, rhs):
    tmp = 0
    for i in range(3):
        tmp += lhs[i] * rhs[i]
    return tmp


def VectDiff(lhs, rhs):
    tmp = lhs.copy()
    for i in range(3):
        tmp[i] -= rhs[i]
    return tmp


def VectScallarMult(lhs, scallar):
    tmp = lhs
    for i in range(3):
        tmp[i] *= scallar
    return tmp


def Dist(lhs, rhs):
    return Norm(VectDiff(lhs, rhs))


def Norm(lhs):
    tmp = Scallar(lhs, lhs)
    return tmp**(0.5)


def VectorProd(lhs, rhs):
    tmp = [0 for i in range(3)]
    for i in range(3):
        tmp[i] = lhs[(i + 1) % 3] * rhs[(i+2) % 3]
        tmp[i] -= lhs[(i + 2) % 3] * rhs[(i+1) % 3]
    return tmp


def ShiftPc(pc, pt):
    res = []
    for p in pc:
        res.append(VectDiff(p, pt))
    return res


def EstimateNormal(pcin, pt):
    pctmp = ShiftPc(pcin, pt)
    normals = []
    n = len(pctmp)
    for i in range(n):
        for j in range(i+1, n):
            tmp = VectorProd(pctmp[i], pctmp[j])
            length = Norm(tmp)
            if (length > 0.0005):
                tmp = VectScallarMult(tmp, 1.0 / length)
                normals.append(tmp)
    a = normals[0]
    sum = a
    for b in normals:
        if (Scallar(a, b) > 0):
            b = VectScallarMult(b, -1.0)
        sum = VectDiff(sum, b)
    length = Norm(sum)
    sum = VectScallarMult(sum, 1.0 / length)
    return sum


def RotateXY(pt, x, y):
    tmp = pt.copy()
    rxy = (x * x + y * y) ** (1/2)
    tmp[0] = 1.0 / rxy * (x * pt[0] + y * pt[1])
    tmp[1] = 1.0 / rxy * (-y * pt[0] + x * pt[1])
    return tmp


def RotateXZ(pt, x, z):
    tmp = pt.copy()
    tmp[0] = x * pt[0] + z * pt[2]
    tmp[2] = -z * pt[0] + x * pt[2]
    return tmp


def RotateVectorToVector(pt, normal):
    x = normal[0]
    y = normal[1]
    z = normal[2]
    rxy = (x*x + y*y)**(0.5)

    tmp = RotateXY(pt, x, y)
    return RotateXZ(tmp, rxy, z)


def RotateToVector(pc, normal):
    n = len(pc)
    x = normal[0]
    y = normal[1]
    rxy = (x*x + y*y)**(0.5)
    if (rxy > 0.005):
        for i in range(n):
            pc[i] = RotateVectorToVector(pc, normal)
    return pc


def Normalize(pc, pt):
    normal = EstimateNormal(pc, pt)
    pc = ShiftPc(pc, pt)
    n = len(pc)
    x = normal[0]
    y = normal[1]
    z = normal[2]
    rxy = (x*x + y*y)**(0.5)
    if (rxy > 0.005):
        for i in range(n):
            tmpx = 1.0 / rxy * (x * pc[i][0] + y * pc[i][1])
            pc[i][1] = 1.0 / rxy * (- y * pc[i][0] + x * pc[i][1])
            pc[i][0] = rxy * tmpx + z * pc[i][2]
            pc[i][2] = - z * tmpx + rxy * pc[i][2]
    return pc


def Pinverse(A):
    A = np.array(A)
    return np.linalg.pinv(A)


def Intepolate(pc, pt):
    pc = Normalize(pc, pt)

    b = [p[2] for p in pc]

    A = []

    for p in pc:
        A.append([])
        for i in range(3):
            for j in range(i + 1):
                A[-1].append((p[1]**j) * (p[0]**(i + 1 - j)))

    Ainv = Pinverse(A)
    return np.dot(Ainv, np.array(b))


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


def AddPointToAx(ax, pt):
    ax.scatter(pt[0], pt[1], pt[2], color='red')


if __name__ == "__main__":

    x = np.linspace(-1, 5, 10)
    y = np.linspace(-1, 5, 10)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    mypc = [[X[i][j], Y[i][j], Z[i][j]] for i in range(10) for j in range(10)]

    mypt = [np.float64(3), np.float64(3), f(3, 3)]

    mypcsh = ShiftPc(mypc, mypt)

    normal = EstimateNormal(mypc, mypt)

    pcnorm = Normalize(mypc, mypt)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xs, ys, zs = X_Y_Z(pcnorm)
    # ax.scatter(xs, ys, zs, color='green')
    a = [1, 1, 1]
    AddPointToAx(ax, [0, 0, 0])
    AddPointToAx(ax, a)
    AddPointToAx(ax, RotateXZ(a, 1 / 5**(0.5), 2 / 5**(0.5)))
    # ax.scatter(normal[0], normal[1], normal[2], color='red')
    ax.set_title('Wireframe Plot')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    plt.show()
