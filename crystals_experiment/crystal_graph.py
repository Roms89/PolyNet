import pandas as pd
import torch
from torch_geometric.data import  Data
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from torch_geometric.data import Dataset



class crystal_graph_dataset(Dataset):

    def __init__(self,
                 root: str,
                 max_d: float,
                 step: float,
                 vor_cut_off: float,
                 targets: pd.DataFrame,
                 ) -> None:

        self.max_d = max_d
        self.step = step
        self.vor_cut_off = vor_cut_off
        self.targets = targets

        self.atomic_nums = [
            8,
            22,
            39,
            40,
            50,
            57,
            58,
            69,
            60,
            62,
            63,
            64,
            66,
            69,
            70,
            72
        ]

        super().__init__(root = root,)

    def process(self):

        idx = 0

        for raw_path in tqdm(self.raw_paths):

            structure = Structure.from_file(raw_path)
            node_feats = self._get_node_feats(structure)
            adj, edge_feats = self._get_edge(structure)
            #file_name = raw_path.split('/')[-1]
            #file_name = file_name.split('.')[0]
            file_name = os.path.splitext(os.path.basename(raw_path))[0]

            print("Checking file:", file_name)
            print("Matches in targets:", self.targets.loc[self.targets[0] == file_name])

            y = self._get_target(file_name)

            coords = self._get_cords(structure)

            data = Data(
                x=node_feats,
                edge_index=adj,
                edge_attr=edge_feats,
                y = y,
                idx = file_name,
                coords = coords,
                weight_monomer = torch.ones(node_feats.shape[0]),
                )

            torch.save(data, osp.join(self.processed_dir, f'{idx}.pt'))
            idx += 1


    def _get_node_feats(self, structure):
        """
        One-hot encodes atomic numbers based on a predefined list.
        Adds an extra slot for low-occurrence or unknown elements.
        """
        # Predefined atomic numbers (example)
        # Should be defined at class level: self.atomic_nums = [...]
        atomic_nums = self.atomic_nums

        # Create index lookup: atomic number → index in one-hot vector
        num_classes = len(atomic_nums) + 1  # +1 for low-occurrence bucket
        index_map = {Z: i for i, Z in enumerate(atomic_nums)}
        other_index = num_classes - 1

        one_hots = []

        for i in range(len(structure)):
            Z = structure[i].specie.number
            vec = np.zeros(num_classes, dtype=float)

            if Z in index_map:
                vec[index_map[Z]] = 1.0
            else:
                # Low occurrence / unknown element
                vec[other_index] = 1.0

            one_hots.append(vec)

        atom_feats = torch.tensor(np.vstack(one_hots), dtype=torch.float)
        return atom_feats


    def _get_edge(self, structure):

        vnn = VoronoiNN(cutoff=self.vor_cut_off,allow_pathological=True,compute_adj_neighbors=False)

        nbr_fea_idx, nbr_fea_t = [], []

        for central_atom in range(len(structure)):

            nbrs = vnn.get_nn_info(structure, central_atom)

            for nbr_info in nbrs: # newer version
                if nbr_info['poly_info']['face_dist']*2 <= self.vor_cut_off:
                    nbr_fea_idx.append([central_atom,nbr_info['site_index']])
                    nbr_fea_t.append(nbr_info['poly_info']['solid_angle'])

        nbr_fea_t = self._GaussianExpansion(vmin=0, vmax=self.max_d, step = self.step, v = np.array(nbr_fea_t))
        nbr_fea_idx = np.array(nbr_fea_idx).transpose()
        nbr_fea = np.array(nbr_fea_t)

        return torch.tensor(nbr_fea_idx), \
            torch.tensor(nbr_fea, dtype=torch.float)


    def _get_target(self, file_name):
        y = self.targets.loc[self.targets[0] == file_name, 1].values
        y = torch.tensor(y, dtype=torch.float).reshape(1)
        return y


    def _get_cords(self, structure):
        coords = []
        for atom in structure:
            coords.append(atom.coords)
        coords = np.vstack(coords)
        return coords

    def _GaussianExpansion(self, vmin, vmax, step, v, var = None):

        assert vmin < vmax
        assert vmax - vmin > step

        filter = np.arange(vmin, vmax+step, step)

        if var is None:
            var = step
        var = var

        return np.exp(-(v[..., np.newaxis] - filter)**2 / var**2)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        material = torch.load(os.path.join(self.processed_dir,
                                f'{idx}.pt'), weights_only=False)
        return material

    @property
    def raw_file_names(self):
        files = os.listdir(os.path.join(self.root, 'raw'))
        return sorted(files)

    @property
    def processed_file_names(self):
        return [f'{idx}.pt' for idx in range(len(self.raw_file_names))]

    def download(self):
        self.raw_file_names()
        raise NotImplementedError
