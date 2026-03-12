import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict






class SimplicialComplex:

    
    def __init__(self, points_3d, k=8, max_edge_length=None):
 
        self.points = points_3d
        self.n_nodes = len(points_3d)
        
    
        self._build_1_simplices(k, max_edge_length)
        self._build_2_simplices()
        
      
        self._build_boundary_operators()
        
        
        self._build_hodge_laplacians()
    
    def _build_1_simplices(self, k, max_edge_length):

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)
        
        edges_set = set()
        edge_lengths = {}
        
        for i in range(self.n_nodes):
            for j in range(1, k+1):  # Skip self (index 0)
                neighbor = indices[i, j]
                dist = distances[i, j]
                
                # Filter by max edge length if specified
                if max_edge_length is not None and dist > max_edge_length:
                    continue
                
                # Use sorted tuple for undirected edges
                edge = tuple(sorted([i, neighbor]))
                if edge not in edges_set:
                    edges_set.add(edge)
                    edge_lengths[edge] = dist
        
        self.edges = list(edges_set)
        self.edge_lengths = edge_lengths
        self.n_edges = len(self.edges)

        self.edge_to_idx = {e: idx for idx, e in enumerate(self.edges)}
    
    def _build_2_simplices(self):

        adj = defaultdict(set)
        for (i, j) in self.edges:
            adj[i].add(j)
            adj[j].add(i)
        
        triangles_set = set()
        
 
        for (i, j) in self.edges:
            # Common neighbors of i and j
            common = adj[i] & adj[j]
            for k in common:
                # Create canonical triangle representation (sorted)
                tri = tuple(sorted([i, j, k]))
                if tri not in triangles_set:
                    # Verify all three edges exist (should be true by construction)
                    e1 = tuple(sorted([i, j]))
                    e2 = tuple(sorted([j, k]))
                    e3 = tuple(sorted([i, k]))
                    if all(e in self.edge_to_idx for e in [e1, e2, e3]):
                        triangles_set.add(tri)
        
        self.triangles = list(triangles_set)
        self.n_triangles = len(self.triangles)
        self.triangle_to_idx = {t: idx for idx, t in enumerate(self.triangles)}
    
    def _build_boundary_operators(self):
  
        B1 = np.zeros((self.n_edges, self.n_nodes))
        for idx, (i, j) in enumerate(self.edges):
            # Convention: edge goes from smaller to larger index
            B1[idx, i] = -1  # Tail (source)
            B1[idx, j] = +1  # Head (target)
        
        self.B1 = torch.FloatTensor(B1)
        

        B2 = np.zeros((self.n_triangles, self.n_edges))
        
        for t_idx, (i, j, k) in enumerate(self.triangles):
            
            e_ij = tuple(sorted([i, j]))
            e_jk = tuple(sorted([j, k]))
            e_ik = tuple(sorted([i, k]))
            
            if e_ij in self.edge_to_idx:
                B2[t_idx, self.edge_to_idx[e_ij]] = +1
            if e_jk in self.edge_to_idx:
                B2[t_idx, self.edge_to_idx[e_jk]] = +1
            if e_ik in self.edge_to_idx:
                B2[t_idx, self.edge_to_idx[e_ik]] = -1
        
        self.B2 = torch.FloatTensor(B2)
    
    def _build_hodge_laplacians(self):
        """
        Build Hodge Laplacians for each simplex dimension.
        """

        self.L0 = self.B1.T @ self.B1
        
 
        if self.n_triangles > 0:
            self.L1 = self.B1 @ self.B1.T + self.B2.T @ self.B2
        else:
            self.L1 = self.B1 @ self.B1.T
        
        
        if self.n_triangles > 0:
            self.L2 = self.B2 @ self.B2.T
        else:
            self.L2 = torch.zeros((0, 0))
        
    
        self.L0_normalized = self._normalize_laplacian(self.L0)
        self.L1_normalized = self._normalize_laplacian(self.L1)
        if self.n_triangles > 0:
            self.L2_normalized = self._normalize_laplacian(self.L2)
        else:
            self.L2_normalized = self.L2
    
    def _normalize_laplacian(self, L):

        D = torch.diag(L).clamp(min=1e-8)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
        return D_inv_sqrt @ L @ D_inv_sqrt
    
    def get_edge_features(self):

        features = []
        for (i, j) in self.edges:
            vec = self.points[j] - self.points[i]
            length = np.linalg.norm(vec)
            vec_norm = vec / (length + 1e-8)
            features.append([length, vec_norm[0], vec_norm[1], vec_norm[2]])
        return torch.FloatTensor(features)
    
    def get_triangle_features(self):

        if self.n_triangles == 0:
            return torch.zeros((0, 4))
        
        features = []
        for (i, j, k) in self.triangles:
            # Two edge vectors of triangle
            v1 = self.points[j] - self.points[i]
            v2 = self.points[k] - self.points[i]
            
            # Cross product gives normal and area
            cross = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross)
            normal = cross / (np.linalg.norm(cross) + 1e-8)
            
            features.append([area, normal[0], normal[1], normal[2]])
        
        return torch.FloatTensor(features)
    
    def get_node_edge_incidence(self):

        A = np.zeros((self.n_nodes, self.n_edges))
        for e_idx, (i, j) in enumerate(self.edges):
            A[i, e_idx] = 1
            A[j, e_idx] = 1
        return torch.FloatTensor(A)
    
    def get_edge_triangle_incidence(self):
    
        if self.n_triangles == 0:
            return torch.zeros((self.n_edges, 0))
        
        A = np.zeros((self.n_edges, self.n_triangles))
        for t_idx, (i, j, k) in enumerate(self.triangles):
            for edge in [(i,j), (j,k), (i,k)]:
                e = tuple(sorted(edge))
                if e in self.edge_to_idx:
                    A[self.edge_to_idx[e], t_idx] = 1
        return torch.FloatTensor(A)


#curvature

def estimate_normal(neighbors_3d):
    """
    Estimate surface normal 
    """
    centroid = neighbors_3d.mean(axis=0)
    centered = neighbors_3d - centroid
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0] 
    return normal / (np.linalg.norm(normal) + 1e-8)


def estimate_curvatures(points_3d, k=12):

    n_points = len(points_3d)
    gaussian_curvature = np.zeros(n_points)
    mean_curvature = np.zeros(n_points)
    
    tree = KDTree(points_3d)
    
    for i in range(n_points):
        # knn
        distances, indices = tree.query(points_3d[i], k=k+1)
        neighbors = points_3d[indices[1:]]  # Exclude self
        
        # Estimate local normal
        normal = estimate_normal(neighbors)
        
        # otheonrmal coordinateframe
        n = normal
    
        t1 = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        t1 = t1 - np.dot(t1, n) * n  # Gram-Schmidt
        t1 = t1 / (np.linalg.norm(t1) + 1e-8)
        t2 = np.cross(n, t1)
        

        neighbors_centered = neighbors - points_3d[i]
        u = neighbors_centered @ t1  
        v = neighbors_centered @ t2  
        h = neighbors_centered @ n   

        if len(u) >= 3:
            A = np.column_stack([u**2, u*v, v**2])
            try:
                coeffs = np.linalg.lstsq(A, h, rcond=None)[0]
                a, b, c = coeffs
                
                # Curvature estimates (first-order approximation)
                H = a + c           # Mean curvature
                K = 4*a*c - b**2    # Gaussian curvature
                
                mean_curvature[i] = H
                gaussian_curvature[i] = K
            except:
                pass
    
    return gaussian_curvature, mean_curvature



def initial_conditions_schnakenberg(points_3d, a=0.048113, b=1.202813):

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    n = len(x)
    
    # Equilibrium values
    u_eq = a + b
    v_eq = b / ((a + b) ** 2)
    
    u_ic = np.full(n, u_eq)
    v_ic = np.full(n, v_eq)
    
    # Add multi-frequency perturbations
    for i in range(1, 6):
        factor = 1.0 / (20 ** i)
        u_ic += factor * np.sin(2*np.pi*i*x) * np.sin(2*np.pi*i*y) * np.sin(2*np.pi*i*z)
        v_ic += factor * np.cos(2*np.pi*i*x) * np.cos(2*np.pi*i*y) * np.cos(2*np.pi*i*z)
    
    return u_ic, v_ic


def initial_conditions_spiral(points_3d):
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    u_ic = np.zeros(len(x))
    v_ic = np.zeros(len(x))
    
    # Different values in different octants
    mask1 = (x > 0) & (y > 0) & (z > 0)
    u_ic[mask1] = 1.0
    v_ic[mask1] = 0.0
    
    mask2 = (x < 0) & (y > 0) & (z > 0)
    u_ic[mask2] = 0.0
    v_ic[mask2] = 1.0
    
    return u_ic, v_ic



class SimplicialConvLayer(nn.Module):
    
    def __init__(self, node_dim, edge_dim, tri_dim, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        self.tri_update = nn.Sequential(
            nn.Linear(tri_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, tri_dim)
        )
        
        self.edge_to_node = nn.Linear(edge_dim, hidden_dim)
        
        self.node_to_edge = nn.Linear(node_dim, hidden_dim)
        
        self.tri_to_edge = nn.Linear(tri_dim, hidden_dim)
        
        self.edge_to_tri = nn.Linear(edge_dim, hidden_dim)
    
    def forward(self, node_feat, edge_feat, tri_feat, sc):

        node_smooth = sc.L0_normalized @ node_feat
        
        # Edge diffusion via L1
        edge_smooth = sc.L1_normalized @ edge_feat
        
        # Triangle diffusion via L2
        if sc.n_triangles > 0 and tri_feat.shape[0] > 0:
            tri_smooth = sc.L2_normalized @ tri_feat
        else:
            tri_smooth = tri_feat
        
       # here is boundary message passing
        edge_msg = self.edge_to_node(edge_feat)  # (E, hidden)
        node_edge_inc = sc.get_node_edge_incidence()  # (N, E)
        degrees = node_edge_inc.sum(dim=1, keepdim=True).clamp(min=1)
        node_from_edges = (node_edge_inc @ edge_msg) / degrees  # (N, hidden)
        
        # node to edge
        node_msg = self.node_to_edge(node_feat)  # (N, hidden)
        # For each edge, aggregate its endpoint node features
        edge_from_nodes = torch.zeros(sc.n_edges, self.hidden_dim)
        for e_idx, (i, j) in enumerate(sc.edges):
            edge_from_nodes[e_idx] = (node_msg[i] + node_msg[j]) / 2
        
        # tri to edge
        if sc.n_triangles > 0 and tri_feat.shape[0] > 0:
            tri_msg = self.tri_to_edge(tri_feat)  # (T, hidden)
            edge_tri_inc = sc.get_edge_triangle_incidence()  # (E, T)
            tri_degrees = edge_tri_inc.sum(dim=1, keepdim=True).clamp(min=1)
            edge_from_tris = (edge_tri_inc @ tri_msg) / tri_degrees  # (E, hidden)
        else:
            edge_from_tris = torch.zeros(sc.n_edges, self.hidden_dim)
        
        # edge to tri
        if sc.n_triangles > 0:
            edge_msg_tri = self.edge_to_tri(edge_feat)  # (E, hidden)
            tri_from_edges = torch.zeros(sc.n_triangles, self.hidden_dim)
            for t_idx, (i, j, k) in enumerate(sc.triangles):
                # Get the three boundary edges
                edge_indices = []
                for edge in [(i,j), (j,k), (i,k)]:
                    e = tuple(sorted(edge))
                    if e in sc.edge_to_idx:
                        edge_indices.append(sc.edge_to_idx[e])
                if edge_indices:
                    tri_from_edges[t_idx] = edge_msg_tri[edge_indices].mean(dim=0)
        else:
            tri_from_edges = torch.zeros(0, self.hidden_dim)
        

        node_combined = torch.cat([node_feat, node_from_edges], dim=-1)
        node_out = node_feat + 0.1 * self.node_update(node_combined)
        
        
        edge_combined = torch.cat([edge_feat, edge_from_nodes + edge_from_tris], dim=-1)
        edge_out = edge_feat + 0.1 * self.edge_update(edge_combined)
        
    
        if sc.n_triangles > 0 and tri_feat.shape[0] > 0:
            tri_combined = torch.cat([tri_feat, tri_from_edges], dim=-1)
            tri_out = tri_feat + 0.1 * self.tri_update(tri_combined)
        else:
            tri_out = tri_feat
        
        # smoothing
        node_out = node_out - 0.05 * node_smooth
        edge_out = edge_out - 0.05 * edge_smooth
        if sc.n_triangles > 0 and tri_feat.shape[0] > 0:
            tri_out = tri_out - 0.05 * tri_smooth
        
        return node_out, edge_out, tri_out


class PISNNModel(nn.Module):

    
    def __init__(self, hidden_dim=32, n_layers=3):
        super().__init__()
        

        self.node_feat_dim = hidden_dim  
        self.edge_feat_dim = 4           
        self.tri_feat_dim = 4            
        

        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        

        self.snn_layers = nn.ModuleList([
            SimplicialConvLayer(hidden_dim, self.edge_feat_dim, self.tri_feat_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.laplacian_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) 
        )
        
        
        self.epsilon = 0.01  
        self.a = 0.1          
        self.b = 0.5          
        self.mu = 0.0001      
        
        self.hidden_dim = hidden_dim
    
    def forward(self, uv, curvatures, sc, edge_feat, tri_feat):
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        
        # Encode node features with physics variables and geometry
        node_input = torch.cat([uv, curvatures], dim=-1)  # (N, 4)
        node_feat = self.encoder(node_input)  # (N, hidden)
        
        for layer in self.snn_layers:
            node_feat, edge_feat, tri_feat = layer(
                node_feat, edge_feat, tri_feat, sc
            )
        
        laplacian_uv = self.laplacian_decoder(node_feat)  # (N, 2)
        laplacian_u = laplacian_uv[:, 0:1]
        laplacian_v = laplacian_uv[:, 1:2]
        
        reaction_u = (self.a - u) * (u - 1.0) * u - v
        reaction_v = self.epsilon * (self.b * u - v)
        du_dt = reaction_u + self.mu * laplacian_u #SNN ~ Laplacian
        dv_dt = reaction_v + self.mu * 0.1 * laplacian_v  # SNN ~ Laplacian
        
        return torch.cat([du_dt, dv_dt], dim=-1)




def generate_ground_truth(uv_init, sc, n_steps=20, dt=0.05):

    n_nodes = uv_init.shape[0]
    
    L = sc.L0.numpy()
    D = np.diag(L).clip(min=1e-8)
    L_norm = L / D.max()  
    
    u = uv_init[:, 0].numpy().copy()
    v = uv_init[:, 1].numpy().copy()
    
    # Physics parameters
    epsilon = 0.01
    a = 0.1
    b = 0.5
    mu = 0.0001
    
    u_seq = [torch.FloatTensor(u).reshape(-1, 1)]
    v_seq = [torch.FloatTensor(v).reshape(-1, 1)]
    
    def dynamics(u, v):
        reaction_u = (a - u) * (u - 1.0) * u - v
        reaction_v = epsilon * (b * u - v)
        laplacian_u = -L_norm @ u
        du_dt = reaction_u + mu * laplacian_u
        dv_dt = reaction_v
        
        return du_dt, dv_dt
    
    for step in range(n_steps - 1):
        k1_u, k1_v = dynamics(u, v)
        k2_u, k2_v = dynamics(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v)
        k3_u, k3_v = dynamics(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v)
        k4_u, k4_v = dynamics(u + dt*k3_u, v + dt*k3_v)
        
        u = u + (dt/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
        u = np.clip(u, -2.0, 2.0)
        v = np.clip(v, -2.0, 2.0)
        
        u_seq.append(torch.FloatTensor(u.copy()).reshape(-1, 1))
        v_seq.append(torch.FloatTensor(v.copy()).reshape(-1, 1))
        
        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{n_steps-1}, u: [{u.min():.3f}, {u.max():.3f}]")
    
    return torch.stack(u_seq), torch.stack(v_seq)


def train_pi_snn(model, uv_init, curvatures, sc, edge_feat, tri_feat,
                 n_epochs=1000, lr=0.001, dt=0.05):
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    u_true_seq, v_true_seq = generate_ground_truth(uv_init, sc, n_steps=50, dt=dt)
    n_steps = len(u_true_seq)
    
    # Train/test split
    n_train = int(0.6 * n_steps)
    u_true_train = u_true_seq[:n_train]
    v_true_train = v_true_seq[:n_train]
    
    print(f"\nGround truth: {n_steps} steps (train: {n_train})")
    print(f"  u range: [{u_true_seq[0].min():.3f}, {u_true_seq[-1].max():.3f}]")
    
    print("\nTraining PI-SNN...")
    loss_history = []
    train_rmse_history = []
    test_rmse_history = []
    best_test_rmse = float('inf')
    patience = 0
    max_patience = 200
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
    
        u_pred = [uv_init[:, 0:1].clone()]
        v_pred = [uv_init[:, 1:2].clone()]
        uv = uv_init.clone()
        
        for step in range(n_train - 1):
            duv_dt = model(uv, curvatures, sc, edge_feat.clone(), tri_feat.clone())
            uv = uv + dt * duv_dt
            uv = torch.clamp(uv, -2.0, 2.0)
            u_pred.append(uv[:, 0:1])
            v_pred.append(uv[:, 1:2])
        
        u_pred_train = torch.stack(u_pred)
        v_pred_train = torch.stack(v_pred)
        
     
        loss = torch.mean((u_pred_train - u_true_train)**2) + \
               torch.mean((v_pred_train - v_true_train)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluate on full sequence
        model.eval()
        with torch.no_grad():
            u_test = [uv_init[:, 0:1].clone()]
            v_test = [uv_init[:, 1:2].clone()]
            uv_test = uv_init.clone()
            
            for step in range(n_steps - 1):
                duv_dt = model(uv_test, curvatures, sc, edge_feat.clone(), tri_feat.clone())
                uv_test = uv_test + dt * duv_dt
                uv_test = torch.clamp(uv_test, -2.0, 2.0)
                u_test.append(uv_test[:, 0:1])
                v_test.append(uv_test[:, 1:2])
            
            u_pred_test = torch.stack(u_test)
            v_pred_test = torch.stack(v_test)
            test_loss = torch.mean((u_pred_test - u_true_seq)**2) + \
                       torch.mean((v_pred_test - v_true_seq)**2)
            test_rmse = torch.sqrt(test_loss).item()
        
        train_rmse = torch.sqrt(loss).item()
        loss_history.append(loss.item())
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)
        
        # Early stopping
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Train RMSE={train_rmse:.6f}, Test RMSE={test_rmse:.6f}")
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return loss_history, train_rmse_history, test_rmse_history, u_true_seq, v_true_seq
