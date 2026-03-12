#class for curvature estimation, will be taking gen_manifold object as input. To compare estimates, 




import numpy as np
from scipy.spatial import cKDTree


class CurvatureEstimator:


    def __init__(self, manifold):
        self._manifold = manifold
        self.gt_torus = None
        self.gt_distorus = None
        self.gt_bunny = None

        mtype = manifold.manifold_type
        if mtype == "torus":
            self.gt_torus = self._gt_torus()
        elif mtype == "distorted_torus":
            self.gt_distorus = self._gt_distorus()
        elif mtype == "bunny":
            self.gt_bunny = self._gt_bunny()


    def _gt_torus(self):

        m = self._manifold
        R, r = m.R, m.r
        u = m._ground_truth_uv[:, 0]

        denom = r * (R + r * np.cos(u))
        K = np.cos(u) / denom
        H = (R + 2 * r * np.cos(u)) / (2 * denom)

        disc = np.maximum(H**2 - K, 0.0)
        k1 = H + np.sqrt(disc)
        k2 = H - np.sqrt(disc)
        return {"K": K, "H": H, "k1": k1, "k2": k2}

    def _gt_distorus(self):
        return self._quadric_curvature(self._manifold._ground_truth)

    def _gt_bunny(self):
        return self._quadric_curvature(self._manifold._ground_truth)


    def estimate(self, points, k_neighbours=15):
        return self._quadric_curvature(points, k_neighbours)

    @staticmethod
    def _quadric_curvature(points, k_neighbours=15):
        #quandratic surface fitting

        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        tree = cKDTree(pts)

        K_arr = np.zeros(n)
        H_arr = np.zeros(n)
        k1_arr = np.zeros(n)
        k2_arr = np.zeros(n)

        k = min(k_neighbours, n - 1)

        normals = np.zeros((n, 3))
        nbr_cache = {}
        for i in range(n):
            dists, nbr_idx = tree.query(pts[i], k=k + 1)
            nbr_cache[i] = nbr_idx
            patch = pts[nbr_idx] - pts[i]
            cov = patch.T @ patch
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]

        normals = CurvatureEstimator._orient_normals(normals, nbr_cache)

        for i in range(n):
            nbr_idx = nbr_cache[i]
            nbr = pts[nbr_idx]
            patch = nbr - pts[i]
            normal = normals[i]

              #rotate
            R = CurvatureEstimator._rotation_to_z(normal)
            patch_rot = (R @ patch.T).T  # (k+1, 3)

            X = patch_rot[1:, 0]
            Y = patch_rot[1:, 1]
            Z = patch_rot[1:, 2]

            #standard regression with least squares sol
            A = np.column_stack([X**2, X * Y, Y**2, X, Y, np.ones(len(X))])
            result = np.linalg.lstsq(A, Z, rcond=None)
            coeffs = result[0]  # [a, b, c, d, e, f]
            a, b, c, d, e, f = coeffs
            fx, fy = d, e
            fxx, fxy, fyy = 2 * a, b, 2 * c
            E = 1 + fx**2
            F = fx * fy
            G = 1 + fy**2
            denom_n = np.sqrt(1 + fx**2 + fy**2)
            L = fxx / denom_n
            M = fxy / denom_n
            N = fyy / denom_n
            det_I = E * G - F**2
            if abs(det_I) < 1e-15:
                continue
            K_val = (L * N - M**2) / det_I
            H_val = (E * N - 2 * F * M + G * L) / (2 * det_I)
            K_arr[i] = K_val
            H_arr[i] = H_val
            disc = max(H_val**2 - K_val, 0.0)
            k1_arr[i] = H_val + np.sqrt(disc)
            k2_arr[i] = H_val - np.sqrt(disc)

        return {"K": K_arr, "H": H_arr, "k1": k1_arr, "k2": k2_arr}

    @staticmethod
    def _orient_normals(normals, nbr_cache):
        from collections import deque

        n = len(normals)
        oriented = np.zeros(n, dtype=bool)
        queue = deque([0])
        oriented[0] = True

        while queue:
            curr = queue.popleft()
            for nb in nbr_cache[curr]:
                if nb == curr or oriented[nb]:
                    continue
                if np.dot(normals[curr], normals[nb]) < 0:
                    normals[nb] = -normals[nb]
                oriented[nb] = True
                queue.append(nb)

        return normals

    @staticmethod
    def _rotation_to_z(normal):
        n = normal / (np.linalg.norm(normal) + 1e-15)
        z = np.array([0.0, 0.0, 1.0])
        dot = np.dot(n, z)
        if abs(dot) > 0.9999:
            if dot > 0:
                return np.eye(3)
            else:
                return np.diag([1.0, -1.0, -1.0])

        v = np.cross(n, z)
        s = np.linalg.norm(v)
        c = dot
        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

        R = np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c))
        return R