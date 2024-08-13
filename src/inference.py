import os
import argparse
import logging
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
import haiku as hk
import jax
import jax.numpy as jnp
import sonnet as snt
from jax2tf import jax2tf
from model import LigandTransformer
import pipeline_batch
import config
import tree

class ModelRunner:
    def __init__(self,config,params=None,is_training=True) -> None:
        self.config=config
        self.params=params
        def _forward_fn(feat):
          module=LigandTransformer(self.config)
          return module(feat,is_training,compute_loss=False)
        self.apply = jax.jit(hk.transform(_forward_fn).apply)
        self.init = jax.jit(hk.transform(_forward_fn).init)
    def init_params(self,feat,random_seed: int=0):
      if not self.params:
        rng=jax.random.PRNGKey(random_seed)
        self.params=hk.data_structures.to_mutable_dict(self.init(rng,feat))
        logging.warning('Initialized parameters randomly')
    def test(self,feat_test,args,random_seed=0):
      mirrored_strategy = tf.distribute.MirroredStrategy()
      data_config=self.config.data
      batch_size=data_config.training.batch_size
      learning_rate=self.config.data.training.learning_rate
      beta_1=self.config.data.training.beta_1
      beta_2=self.config.data.training.beta_2
      class JaxModule(snt.Module):
        def __init__(self, params, apply_fn, name=None):
          def create_variable(path, value):
            name = '/'.join(map(str, path)).replace('~', '_')
            return tf.Variable(value, name=name)
          super().__init__(name=name)
          self._params = tree.map_structure_with_path(create_variable, params)
          self._apply = jax2tf.convert(lambda p, x: apply_fn(p, jax.random.PRNGKey(random_seed), x))
          self._apply = tf.autograph.experimental.do_not_convert(self._apply)
        def __call__(self,feat):
          return self._apply(self._params,feat)
      
      with mirrored_strategy.scope():
        net= JaxModule(self.params,self.apply)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         global_clipnorm=0.1
                                         )
      @tf.function(experimental_compile=True, autograph=False)
      def test_step(feat):
        ret=net(feat)
        return ret
      checkpoint=tf.train.Checkpoint(optimize=opt,model=net)
      save_folder=args.ckpt_path
      checkpoint.restore(tf.train.latest_checkpoint(save_folder))

      true_affinities=np.array([])
      calculated_affinities=np.array([])                      
      ids=np.array([])
      for feat in mirrored_strategy.experimental_distribute_dataset(feat_test.batch(batch_size)):
        id_i=feat['id']
        if id_i.shape[0]!=batch_size:continue
        ret = test_step(feat) 
        affinity=feat['affinity']
        calculated_affinity=ret['affinity']['ret']
        true_affinities=np.append(true_affinities,affinity)
        calculated_affinities=np.append(calculated_affinities,calculated_affinity)
        ids=np.concatenate((ids,id_i),axis=0)
      with open(f"{args.output_dir}/{args.output_csv}",'w') as f:
        for i,(id,y)in enumerate(zip(ids,calculated_affinities)):
          f.write(f"{id},{y},\n")
      return 
    
def run(args):
  data_paths={'test':args.mol_library_pickle}
  protein_data=pd.read_csv(args.protein_csv)
  if len(protein_data)>1:
    raise ValueError("Now only support one protein.")
  protein_row=protein_data.iloc[0]
  protein_feature_prefix=protein_row['protein_feat_prefix']
  resi_num=int(protein_row['resi_num'])

  configure=config.model_config()
  configure.data.training.crop_size=resi_num
  data_config=configure.data
  batch_size=data_config.training.batch_size
  gpu_count=len(tf.config.list_physical_devices('GPU'))
  crop_size=data_config.training.crop_size

  generator=pipeline_batch.DataGenerator(data_paths,protein_feature_prefix,data_config,resi_num)

  data_config.training.atom_crop_size=generator.max_atom_num
  atom_crop_size=data_config.training.atom_crop_size
  single_channels=data_config.single_channels
  pair_channels=data_config.pair_channels
  atom_embeddings_channels=data_config.atom_embeddings_channels
  edge_embeddings_channels=data_config.edge_embeddings_channels
  msa_act=np.random.random((int(batch_size/gpu_count),crop_size,single_channels))
  struc_act=np.random.random((int(batch_size/gpu_count),crop_size,single_channels))
  msa_mask=np.ones((int(batch_size/gpu_count),crop_size))
  pair_mask=np.ones((int(batch_size/gpu_count),crop_size,crop_size))
  pair_act=np.random.random((int(batch_size/gpu_count),crop_size,crop_size,pair_channels))
  atom_act=np.random.random((int(batch_size/gpu_count),atom_crop_size,atom_embeddings_channels))
  atom_mask=np.ones((int(batch_size/gpu_count),atom_crop_size))
  edge_mask=np.ones((int(batch_size/gpu_count),atom_crop_size,atom_crop_size))
  edge_act=np.random.random((int(batch_size/gpu_count),atom_crop_size,atom_crop_size,edge_embeddings_channels))
  distmat=np.ones((int(batch_size/gpu_count),crop_size+atom_crop_size,crop_size+atom_crop_size))
  distmat_mask=np.ones((int(batch_size/gpu_count),crop_size+atom_crop_size))
  affinity=np.random.random((int(batch_size/gpu_count),1))
  resi_num=np.ones(int(batch_size/gpu_count))*crop_size
  atom_num=np.ones(int(batch_size/gpu_count))*atom_crop_size

  feat={'msa_act':msa_act,'struc_act':struc_act,'msa_mask':msa_mask,'pair_act':pair_act,'pair_mask':pair_mask,
      'atom_act':atom_act,'atom_mask':atom_mask,'edge_act':edge_act,'edge_mask':edge_mask,
      'affinity':affinity,'distmat':distmat,'distmat_mask':distmat_mask,'resi_num':resi_num,'atom_num':atom_num}
  model_runner=ModelRunner(config=configure,is_training=False)
  model_runner.init_params(feat)
  del feat

  generator_test=functools.partial(
        generator.generate,
        data_name='test',
        shuffle=False)
  feat_test=tf.data.Dataset.from_generator(generator=generator_test,output_types=generator.output_types,output_shapes=generator.output_shapes)  
  model_runner.test(feat_test,args)

if __name__=='__main__':
  import os
  parser=argparse.ArgumentParser()
  parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory",
        required=True,
    )
  parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to the output csv file",
    )
  parser.add_argument(
        "--protein_csv",
        type=str,
        help="Path to the protein csv file, now only support one protein.",
    )
  parser.add_argument(
        "--mol_library_pickle",
        type=str,
        help="Path to the pickle file containing the molecule library",
    )
  parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the checkpoint",
    )
  args=parser.parse_args()
  run(args)