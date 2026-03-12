import sympy as sp


def SymbolicRiemanFromMetric(g, x, pt=[0, 0]):
    detg = sp.simplify(g[0][0] * g[1][1] - g[0][1]**2)

    ginv = [[g[1][1]/detg, -g[1][0]/detg], [-g[0][1]/detg, g[0][0] / detg]]

    Gamma = [
        [
            [
                sp.simplify(
                    0.5 * (
                        ginv[i][0] * (sp.diff(g[0][k], x[l]) +
                                      sp.diff(g[0][l], x[k]) -
                                      sp.diff(g[l][k], x[0]))
                        +
                        ginv[i][1] * (sp.diff(g[1][k], x[l]) +
                                      sp.diff(g[1][l], x[k]) -
                                      sp.diff(g[l][k], x[1]))
                    )
                )
                for l in range(2)
            ]
            for k in range(2)
        ]
        for i in range(2)
    ]

    R = [
        [
            [
                [
                    sp.simplify((
                                sp.diff(Gamma[i][j][m], x[k]) +
                                sp.diff(Gamma[i][j][k], x[m]) +
                                Gamma[i][0][k] * Gamma[0][j][m] +
                                Gamma[i][1][k] * Gamma[1][j][m] -
                                Gamma[i][0][m] * Gamma[0][j][k] -
                                Gamma[i][1][m] * Gamma[1][j][k]

                                ).subs(x[0], pt[0]).subs(x[1], pt[1]))




                    for m in range(2)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]
    return R


def ReimanByCoef(Coff):
    x = [sp.symbols('x'), sp.symbols('y')]
    f = 0
    C = []
    cnt = 0
    for i in range(1, 4):
        C.append([])
        for j in range(i + 1):
            C[-1].append(float(Coff[cnt]))
            cnt += 1
            f += C[-1][-1] * x[0] ** (i-j) * x[1]**(j)

    g = [[1 + sp.diff(f, x[0])**2, sp.diff(f, x[0]) * sp.diff(f, x[1])],
         [sp.diff(f, x[0]) * sp.diff(f, x[1]), 1 + sp.diff(f, x[1])**2]]
    return SymbolicRiemanFromMetric(g, x)
