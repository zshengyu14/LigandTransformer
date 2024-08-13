import os
import pickle
import traceback
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
import multiprocessing as mp
import argparse
from tqdm import tqdm
import logging
import traceback

from datasets import allowable_features
from molecule_gnn_model import GNN
from torch_geometric.data import DataLoader

# To avoid deadlock in the multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # Atoms: two features - atom type and chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # Bonds: two features - bond type and bond direction
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data

def process_csv_row(args, row):
    index = int(row[args.id_col])
    try:
        smi = row[args.smiles_col]
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, randomSeed=0)
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
        elif res == -1:
            mol_tmp = Chem.MolFromSmiles(smi)
            AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=0)
            mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol_tmp)
                mol = mol_tmp
            except Exception:
                pass
        data = mol_to_graph_data_obj_simple_3D(mol)
        data.mol_id = torch.tensor([index])
        data.id = torch.tensor([index])
        return data
    except Exception as e:
        traceback.print_exc()
        return None

def process_mol(args, mol, index):
    try:
        if args.addH:
            mol = AllChem.AddHs(mol)
        data = mol_to_graph_data_obj_simple_3D(mol)
        data.mol_id = torch.tensor([index])
        data.id = torch.tensor([index])
        return data
    except Exception as e:
        traceback.print_exc()
        return None

def process_mol_file(args, file):
    try:
        index = int(file.split('.')[0])
        if file.endswith('.sdf'):
            mol = Chem.MolFromMolFile(os.path.join(args.input_dir, file), removeHs=False)
        elif file.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(os.path.join(args.input_dir, file), removeHs=False)
        elif file.endswith('.mol2'):
            mol = Chem.MolFromMol2File(os.path.join(args.input_dir, file), removeHs=False)
        if args.addH:
            mol = AllChem.AddHs(mol)
        data = process_mol(args, mol, index)
        return data
    except Exception as e:
        traceback.print_exc()
        return None
    
def handle_future(future):
    try:
        result = future.get(timeout=60)  # Timeout after 60 seconds
        return result
    except mp.TimeoutError:
        logging.warning("A process timed out.")
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
        traceback.print_exc()
    return None

class MyDataSet(InMemoryDataset):
    """
    Support three types of data input:
    1. CSV file with columns: 'id', 'SMILES',
    2. Directory with .sdf or .pdb or .mol2 files
    3. A large .sdf file with id label
    """
    def __init__(self, args):
        self.args = args
        super().__init__(args.output_dir, None, None, None)
        self.data, self.slices = torch.load(f"{args.output_dir}/processed.pt")
    
    @property
    def processed_file_names(self):
        return [os.path.join(self.args.output_dir, 'processed.pt')]

    def process(self):
        args = self.args
        if not args.force_reprocess and os.path.exists(f"{args.output_dir}/processed.pt"):
            self.data, self.slices = torch.load(f"{args.output_dir}/processed.pt")
            return
        
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError('Data directory not found')
        
        data_list = []

        if args.input_dir.endswith('.csv'):
            logging.info(f"Processing CSV file: {args.input_dir}")
            data_csv = pd.read_csv(args.input_dir)
            tasks = [(args, row) for _, row in data_csv.iterrows()]
            process_function = process_csv_row

        elif os.path.isdir(args.input_dir):
            logging.info(f"Processing directory: {args.input_dir}")
            files = os.listdir(args.input_dir)
            tasks = [(args, file) for file in files]
            process_function = process_mol_file

        elif args.input_dir.endswith('.sdf'):
            logging.info(f"Processing SDF file: {args.input_dir}")
            suppl = Chem.SDMolSupplier(args.input_dir, removeHs=False)
            tasks = []
            for i, mol in enumerate(suppl):
                try:
                    index = int(mol.GetProp(args.id_col))
                    tasks.append((args, mol, index))
                except Exception as e:
                    logging.warning(f"Error processing molecule {i}: {e}")
                    traceback.print_exc()
            process_function = process_mol

        else:
            logging.error("Unsupported input file type.")

        if args.use_multiprocessing:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=mp.cpu_count()) as pool:
                result_futures = [pool.apply_async(process_function, task) for task in tasks]
                for future in result_futures:
                    result = handle_future(future)
                    if result is not None:
                        data_list.append(result)
        else:
            for task in tqdm(tasks):
                result = process_function(*task)
                if result is not None:
                    data_list.append(result)

        data_list = [data for data in data_list if data is not None]
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), f"{args.output_dir}/processed.pt")

def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MyDataSet(args)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Set up model
    print(f"Total batches to process: {len(train_loader)}")
    molecule_model = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.0, gnn_type='gin')
    molecule_model.to(device)
    molecule_model.load_state_dict(torch.load("/rds/project/rds-a1NGKrlJtrw/dyna_mol/pretraining_model.pth", map_location=device))
    
    if args.pickled_file:
        data_collect={}
    for count, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing data"):
        try:
            id = str(batch.to_data_list()[0]['id'].numpy())
            if not args.force_reprocess and os.path.exists(f"{args.output_dir}/{id}.pickle"):
                continue
            batch.to(device)
            node_representation, eb, edge_attr = molecule_model(batch.x, batch.edge_index, batch.edge_attr)
            data = batch.cpu().to_data_list()[0]
            data = {
                'edge_attr': data.edge_attr.numpy(),
                'edge_index': data.edge_index.numpy(),
                'id': data.id.numpy(),
                'positions': data.positions.numpy(),
                'x': data.x.numpy(),
                'node_representation': node_representation.cpu().detach().numpy(),
                'edge_embeddings': eb.cpu().detach().numpy()
            }
            if not args.pickled_file:
                with open(f"{args.output_dir}/{id}.pickle", 'wb') as wf:
                    pickle.dump(data, wf)
            else:
                data_collect[id] = data
            del batch, node_representation, eb, edge_attr
        except Exception as e:
            traceback.print_exc()
            continue
    if args.pickled_file:
        with open(f"{args.output_dir}/{args.pickled_file}", 'wb') as wf:
            pickle.dump(data_collect, wf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='The input data directory or file')
    parser.add_argument('--id_col', type=str, default='id', 
                        help='The column name of the id')
    parser.add_argument('--smiles_col', type=str, default='SMILES', 
                        help='The column name of the smiles')
    parser.add_argument('--addH', action='store_true', default=True, 
                        help='Add hydrogen atoms to the molecule')
    parser.add_argument('--force_reprocess', action='store_true', default=False, 
                        help='Force to reprocess the data')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='The directory to save the processed data')
    parser.add_argument('--pickled_file', type=str, default=None,
                        help='The pickled file to save all the processed data, if not set, files will be saved seperately.')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='Use multiprocessing to process the data')

    args = parser.parse_args()
    run(args)
