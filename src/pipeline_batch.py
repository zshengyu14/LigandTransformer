import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from time import time

def read_picklefile(filename):
  return pickle.load(open(filename,'rb'))

class DataGenerator():
  def __init__(self,data_paths,protein_feature_prefix,data_config,resi_num):
    print("Starting loading raw data")
    self.data={}
    crop_size=data_config.training.crop_size
    atom_crop_size=data_config.training.atom_crop_size
    pair_channels=data_config.pair_channels
    single_channels=data_config.single_channels
    atom_embeddings_channels=data_config.atom_embeddings_channels
    edge_embeddings_channels=data_config.edge_embeddings_channels
    protein_feats={}
    mol_feats={}
    self.max_atom_num=0
    for data_name,data_path in data_paths.items():
      mol_datas=pickle.load(open(data_path,'rb'))
      pair=np.load(protein_feature_prefix+'_pair_repr_1_model_3.npy')
      single=np.load(protein_feature_prefix+'_single_repr_1_model_3.npy')
      struc=np.load(protein_feature_prefix+'_structure_repr_1_model_3.npy')
      protein_feat={'pair':pair,'single':single,'struc':struc,'resi_num':resi_num}
      protein_feats[0]=protein_feat
      for mol in mol_datas.keys():
        if mol in mol_feats.keys(): continue
        try:
          mol_data=mol_datas[mol]
          mol_feat={'mol_data':mol_data}
          atom_repr=mol_data['node_representation']
          atom_num=atom_repr.shape[0]
          if atom_num>atom_crop_size: continue
          if atom_num>self.max_atom_num: self.max_atom_num=atom_num
          mol_feats[mol]=mol_feat
        except:
          pass
          #print('Error cannot open mol',mol)
      ids=list(mol_feats.keys())
      data=pd.DataFrame({'id':ids})
      print(f'Loaded {len(mol_feats)} mols in {data_path}')
      self.data[data_name]=data

    self.protein_feats=protein_feats
    self.mol_feats=mol_feats

    self.output_types= {'pair_act':np.float32,'msa_act':np.float32,'struc_act':np.float32,'atom_act':np.float32,'edge_act':np.float32,
      'msa_mask':np.int8,'pair_mask':np.int8,'atom_mask':np.int8,'edge_mask':np.int8,
      'resi_num':np.int16,'atom_num':np.int16,'affinity':np.float32,'distmat':np.float32,'distmat_mask':np.int8,'id':np.int64,'data_type':np.int8,'mask':{'msa':np.int8,'atom':np.int8,'distmat':np.int8}}

    self.output_shapes={'pair_act':[crop_size,crop_size,pair_channels],'msa_act':[crop_size,single_channels],'struc_act':[crop_size,single_channels],'atom_act':[atom_crop_size,atom_embeddings_channels],'edge_act':[atom_crop_size,atom_crop_size,edge_embeddings_channels],
      'msa_mask':[crop_size],'pair_mask':[crop_size,crop_size],'atom_mask':[atom_crop_size],'edge_mask':[atom_crop_size,atom_crop_size],
      'resi_num':[],'atom_num':[],'affinity':[],'distmat':[crop_size+atom_crop_size,crop_size+atom_crop_size],'distmat_mask':[crop_size+atom_crop_size],'id':[],'data_type':[],
      'mask':{'msa':[crop_size],'atom':[atom_crop_size],'distmat':[crop_size+atom_crop_size]}}
    self.data_config=data_config

  def generate(self,data_name,shuffle=False):
    data_config=self.data_config
    crop_size=data_config.training.crop_size
    atom_crop_size=data_config.training.atom_crop_size
    atom_embeddings_channels=data_config.atom_embeddings_channels
    edge_embeddings_channels=data_config.edge_embeddings_channels
    batch_size=data_config.training.batch_size
    data=self.data[data_name]
    if shuffle:
      t=time()
      print(f'Random seed for shuffle:{t}')
      data=data.sample(frac = 1,random_state=int(t))
    data_len=int(len(data)//batch_size)*batch_size
    data=data.iloc[:data_len]
    for i,row in data.iterrows():
      try:
        seq_id=0
        mol=row['id']
        id=mol
        if mol not in self.mol_feats.keys():
          continue
        if seq_id not in self.protein_feats.keys():
          continue
        protein_feat=self.protein_feats[seq_id]
        single=protein_feat['single']
        struc=protein_feat['struc']
        pair=protein_feat['pair']
        resi_num=protein_feat['resi_num']
        mol_data=self.mol_feats[mol]['mol_data']

        affinity = 0.0
        data_type = 0
        length=single.shape[0]
        atom_repr=mol_data['node_representation']
        atom_num=atom_repr.shape[0]
        edge_index=mol_data['edge_index']
        edge_repr_raw=mol_data['edge_embeddings']
        edge_repr=np.zeros((atom_num,atom_num,edge_embeddings_channels))
        edge_repr= np.asanyarray(edge_repr,dtype=np.float32)

        protein_dists=np.zeros((resi_num,resi_num))
        mol_dists=np.zeros((atom_num,atom_num))
        distmat=np.zeros((resi_num,atom_num))
        protein_dist_mask=np.zeros((resi_num))
        mol_dist_mask=np.zeros((atom_num))

        for i in range(edge_index.shape[0]):
          edge_repr[edge_index[0][i],edge_index[1][i]]+=edge_repr_raw[i]
        for i in range(atom_num):
          edge_repr[i][i]+=edge_repr_raw[-atom_num+i]

        if atom_num>atom_crop_size: 
          continue
        if resi_num > crop_size:
          continue
        if length > crop_size:
          pair=pair[:crop_size,:crop_size,:]
          single=single[:crop_size,:]
          struc=struc[:crop_size,:]
        length=single.shape[0]
        
        msa_mask=np.ones(resi_num)
        pair_mask=np.ones((resi_num,resi_num))
        atom_mask=np.ones(atom_num)
        edge_mask=np.ones((atom_num,atom_num))

        pair_mask=np.pad(pair_mask,((0,crop_size-resi_num),(0,crop_size-resi_num)),'constant',constant_values=(0,0))
        msa_mask=np.pad(msa_mask,(0,crop_size-resi_num),'constant',constant_values=(0,0))
        edge_mask=np.pad(edge_mask,((0,atom_crop_size-atom_num),(0,atom_crop_size-atom_num)),'constant',constant_values=(0,0))
        atom_mask=np.pad(atom_mask,(0,atom_crop_size-atom_num),'constant',constant_values=(0,0))
        protein_dist_mask=np.pad(protein_dist_mask,(0,crop_size-resi_num),'constant',constant_values=(0,0))
        mol_dist_mask=np.pad(mol_dist_mask,(0,atom_crop_size-atom_num),'constant',constant_values=(0,0))
        distmat_mask=np.concatenate((protein_dist_mask,mol_dist_mask))

        single=np.pad(single,((0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
        struc=np.pad(struc,((0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
        pair=np.pad(pair,((0,crop_size-length),(0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
        atom_repr=np.pad(atom_repr,((0,atom_crop_size-atom_num),(0,0)),'constant',constant_values=(0.0,0.0))
        edge_repr=np.pad(edge_repr,((0,atom_crop_size-atom_num),(0,atom_crop_size-atom_num),(0,0)),'constant',constant_values=(0.0,0.0))
        protein_dists=np.pad(protein_dists,((0,crop_size-resi_num),(0,crop_size-resi_num)),'constant',constant_values=(0,0))
        mol_dists=np.pad(mol_dists,((0,atom_crop_size-atom_num),(0,atom_crop_size-atom_num)),'constant',constant_values=(0,0))
        distmat=np.pad(distmat,((0,crop_size-resi_num),(0,atom_crop_size-atom_num)),'constant',constant_values=(0,0))

        
        protein_dists=np.concatenate((protein_dists,distmat),axis=1)
        mol_dists=np.concatenate((np.transpose(distmat),mol_dists),axis=1)
        distmat=np.concatenate((protein_dists,mol_dists),axis=0)

        pair_mask= np.asanyarray(pair_mask,dtype=np.int8)
        msa_mask= np.asanyarray(msa_mask,dtype=np.int8)    
        edge_mask= np.asanyarray(edge_mask,dtype=np.int8)
        atom_mask= np.asanyarray(atom_mask,dtype=np.int8)
        distmat_mask= np.asanyarray(distmat_mask,dtype=np.int8)
        data_type= np.asanyarray(data_type,dtype=np.int8)

        single= np.asanyarray(single,dtype=np.float32)
        struc= np.asanyarray(struc,dtype=np.float32)
        pair= np.asanyarray(pair,dtype=np.float32)    
        atom_repr= np.asanyarray(atom_repr,dtype=np.float32)
        edge_repr= np.asanyarray(edge_repr,dtype=np.float32)
        distmat= np.asanyarray(distmat,dtype=np.float32)
        affinity= np.asanyarray(affinity,dtype=np.float32)
        id=int(id[1:-1])
        id=np.asanyarray(id,dtype=np.int64)
        #geneid=np.asanyarray(geneid,dtype=np.float32)
        assert single.shape==(crop_size,data_config.single_channels)
        assert struc.shape==(crop_size,data_config.single_channels)
        assert pair.shape==(crop_size,crop_size,data_config.pair_channels)
        assert atom_repr.shape==(atom_crop_size,atom_embeddings_channels)
        assert edge_repr.shape==(atom_crop_size,atom_crop_size,edge_embeddings_channels)
        assert distmat_mask.shape==(crop_size+atom_crop_size,)
        assert distmat.shape==(crop_size+atom_crop_size,crop_size+atom_crop_size)
        assert msa_mask.shape==(crop_size,)
        assert atom_mask.shape==(atom_crop_size,)
        assert pair_mask.shape==(crop_size,crop_size)
        assert edge_mask.shape==(atom_crop_size,atom_crop_size)
        mask={'msa':msa_mask,'atom':atom_mask,'distmat':distmat_mask}
        feat={'pair_act':pair,'msa_act':single,'struc_act':struc,'atom_act':atom_repr,'edge_act':edge_repr,
        'msa_mask':msa_mask,'pair_mask':pair_mask,'atom_mask':atom_mask,'edge_mask':edge_mask,
        'resi_num':resi_num,'atom_num':atom_num,'affinity':affinity,'distmat':distmat,'distmat_mask':distmat_mask,'id':id,'data_type':data_type,'mask':mask}
        yield feat
      except Exception as e:
        print("Error in generating data",row['id'],e)
        continue
