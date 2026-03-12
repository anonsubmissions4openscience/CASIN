#this will generate torus, distorted torus and bunny. 

#TODO add some more manifolds

import numpy as np
import gudhi.datasets.remote as remote
import gudhi

#by default ground truth I give 10k points. 
N_GROUND_TRUTH = 10000

class gen_manifold:
    def __init__(self):
        self.manifold_type = None
        self.ground_truth = None
        self._ground_truth_uv = None  
        self.R = 2
        self.r = 1

    def gen_torus(self, R=2.0, r=1.0):
        self.manifold_type = "torus"
        self.R, self.r = R, r
        u = np.random.uniform(0, 2 * np.pi, N_GROUND_TRUTH)
        v = np.random.uniform(0, 2 * np.pi, N_GROUND_TRUTH)
        x = (R + r * np.cos(u)) * np.cos(v)
        y = (R + r * np.cos(u)) * np.sin(v)
        z = r * np.sin(u)
        self.ground_truth = np.column_stack([x, y, z])
        self._ground_truth_uv = np.column_stack([u, v])
        return self

    def gen_distorus(self, R=2.0, r=1.0, distortion=None, strength=0.3):
        self.manifold_type = "distorted_torus"
        self.R, self.r = R, r
        if distortion is None:
            def distortion(x, y, z):
                return (
                    x + strength * z * y,
                    y + strength * z * x,
                    z + strength * (x**2 - y**2),
                )
        u = np.random.uniform(0, 2 * np.pi, N_GROUND_TRUTH)
        v = np.random.uniform(0, 2 * np.pi, N_GROUND_TRUTH)
        x0 = (R + r * np.cos(u)) * np.cos(v)
        y0 = (R + r * np.cos(u)) * np.sin(v)
        z0 = r * np.sin(u)
        x, y, z = distortion(x0, y0, z0)
        self.ground_truth = np.column_stack([x, y, z])
        self._ground_truth_uv = np.column_stack([u, v])
        return self

    def gen_bunny(self, file_path=None, accept_license=True):
        self.manifold_type = "bunny"
        self.ground_truth = remote.fetch_bunny(
            file_path=file_path, accept_license=accept_license
        )
        self._ground_truth_uv = None
        return self

    def sample(self, num_points):
        if self.ground_truth is None:
            raise RuntimeError("call gen_torus, gen_distorus, or gen_bunny first.")
        if num_points >= len(self.ground_truth):
            return self.ground_truth.copy()
        idx = np.random.choice(len(self.ground_truth), num_points, replace=False)
        return self.ground_truth[idx]