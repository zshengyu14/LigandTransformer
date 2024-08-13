import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import functools
from alphafold.model import mapping
from alphafold.model import prng
from alphafold.model import utils
from alphafold.model import layer_stack

def glorot_uniform():
  return hk.initializers.VarianceScaling(scale=1.0,
                                         mode='fan_avg',
                                         distribution='uniform')

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss)
def mean_absolute_error(rets,labels):
  loss =  jnp.mean(jnp.absolute(labels-rets),axis=-1)
  return jnp.asarray(loss)
def sigmoid_cross_entropy(logits, labels):
  """Computes sigmoid cross entropy given logits and multiple class labels."""
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  loss = -labels * log_p - (1. - labels) * log_not_p
  return jnp.asarray(loss)

def calculate_r_square(y_true,y_pred):
  return 1- np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)

def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
  """Applies dropout to a tensor."""
  if is_training and rate != 0.0:
    shape = list(tensor.shape)
    if broadcast_dim is not None:
      shape[broadcast_dim] = 1
    keep_rate = 1.0 - rate
    random_int=jnp.ravel(tensor)[0]
    rng=jax.random.fold_in(safe_key.get(),jnp.asarray(jnp.absolute((1.0/(random_int+1e-3)+random_int)*1e5),dtype=jnp.int32))
    keep = jax.random.bernoulli(rng, keep_rate, shape=shape)
    return keep * tensor / keep_rate
  else:
    return tensor

def dropout_wrapper(module,
                    input_act,
                    mask,
                    safe_key,
                    global_config,
                    output_act=None,
                    is_training=True,
                    **kwargs):
  """Applies module + dropout + residual update."""
  if output_act is None:
    output_act = input_act

  gc = global_config
  residual = module(input_act, mask, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

  if module.config.shared_dropout:
      broadcast_dim = 0
  else:
    broadcast_dim = None

  residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)

  new_act = output_act + residual

  return new_act

def dropout_wrapper_pair(module,
                    input_single,
                    input_pair,
                    mask_single,
                    safe_key,
                    global_config,
                    is_training=True,
                    **kwargs):
  """Applies module + dropout + residual update."""
  gc = global_config
  residual_single,residual_pair = module(input_single, mask_single, input_pair, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate
  _, *safe_subkey = safe_key.split(3)
  safe_subkey = iter(safe_subkey)
  if module.config.shared_dropout:
      broadcast_dim = 0
  else:
    broadcast_dim = None

  residual_single = apply_dropout(tensor=residual_single,
                           safe_key=next(safe_subkey),
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)
  residual_pair = apply_dropout(tensor=residual_pair,
                           safe_key=next(safe_subkey),
                           rate=dropout_rate,
                           is_training=is_training,
                           broadcast_dim=broadcast_dim)
  new_single = input_single + residual_single
  new_pair = input_pair + residual_pair

  return new_single,new_pair

def _calculate_value_from_logits(logits: jnp.ndarray,bin_centers: jnp.ndarray):
  probs = jax.nn.softmax(logits, axis=-1)
  predicted_value = jnp.sum(probs * jnp.squeeze(bin_centers), axis=-1)
  return predicted_value


class Linear(hk.Module):
  def __init__(self,
               num_output: int,
               initializer: str = 'linear',
               use_bias: bool = True,
               bias_init: float = 0.,
               name: str = 'linear'):
    super().__init__(name=name)
    self.num_output = num_output
    self.initializer = initializer
    self.use_bias = use_bias
    self.bias_init = bias_init

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Arguments:
        inputs: [batch, ..., num_input] input tensor. 
    Returns:
        output: [batch, ..., num_output] output tensor.
    """
    n_channels = int(inputs.shape[-1])

    weight_shape = [n_channels, self.num_output]
    if self.initializer == 'linear':
      weight_init = hk.initializers.VarianceScaling(mode='fan_in', scale=1.)
    elif self.initializer == 'relu':
      weight_init = hk.initializers.VarianceScaling(mode='fan_in', scale=2.)
    elif self.initializer == 'zeros':
      weight_init = hk.initializers.Constant(0.0)

    weights = hk.get_parameter('weights', weight_shape, inputs.dtype,
                               weight_init)

    # this is equivalent to einsum('...c,cd->...d', inputs, weights)
    # but turns out to be slightly faster
    inputs = jnp.swapaxes(inputs, -1, -2)
    output = jnp.einsum('...cb,cd->...db', inputs, weights)
    output = jnp.swapaxes(output, -1, -2)

    if self.use_bias:
      bias = hk.get_parameter('bias', [self.num_output], inputs.dtype,
                              hk.initializers.Constant(self.bias_init))
      output += bias

    return output

class Transition(hk.Module):
  """Transition layer.

  Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
  Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
  """

  def __init__(self, config, global_config, name='transition_block'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask, is_training=True, output_dim=None, with_norm=True):
    """Builds Transition module.

    Arguments:
      act: A tensor of queries of size [batch_size, N_res, N_channel].
      mask: A tensor denoting the mask of size [batch_size, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      A float32 tensor of size [batch_size, N_res, output_dim].
    """
    nc = act.shape[-1]
    if output_dim is None:
      output_dim=nc

    num_intermediate = int(nc * self.config.num_intermediate_factor)
    mask = jnp.expand_dims(mask, axis=-1)

    if with_norm:
      act = hk.LayerNorm(
          axis=[-1],
          create_scale=True,
          create_offset=True,
          name='input_layer_norm')(
              act)

    transition_module = hk.Sequential([
        Linear(
            num_intermediate,
            initializer='relu',
            name='transition1'), jax.nn.relu,
        Linear(
            output_dim,
            initializer=utils.final_init(self.global_config),
            name='transition2')
    ])

    act = mapping.inference_subbatch(
        transition_module,
        self.global_config.subbatch_size,
        batched_args=[act],
        nonbatched_args=[],
        low_memory=not is_training)

    return act

class mAttention(hk.Module):
  def __init__(self, config, global_config, output_dim, name='mAttention'):
      super().__init__(name=name)
      self.config=config
      self.global_config=global_config
      self.output_dim=output_dim
  def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
    """
    Arguments:
        q_data: [batch, N_res, c_q] query data.
        m_data: [batch, N_res, c_m] memory data.
        bias: [batch, N_res, N_res] bias.
        nonbatched_bias: [N_res, N_res] nonbatched bias.
    Returns:
        output: [batch, N_res, c_o] output.
        logits_update: [batch, N_res, N_res, num_head] logits_update to pair representation.
    """

    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head
    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], num_head, key_dim),
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], num_head, value_dim),
        init=glorot_uniform())
    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    logits_update= jnp.einsum('bqhc,bkhc->bhqk', q, k)
    logits = logits_update + bias
    if nonbatched_bias is not None:
      logits += jnp.expand_dims(nonbatched_bias, axis=0)
    weights = jax.nn.softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                              gating_weights) + gating_bias

      gate_values = jax.nn.sigmoid(gate_values)

      weighted_avg *= gate_values

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        init=init)
    o_bias = hk.get_parameter('output_b', shape=(self.output_dim,),
                              init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    return output,logits_update

class AttentionWithPairBias(hk.Module):
  def __init__(self, config, global_config,
               name='msa_row_attention_with_pair_bias'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
              msa_act,
              msa_mask,
              pair_act,
              is_training=False):
    """
    Arguments:
      msa_act: [batch, N_res, c_m] MSA representation.
      msa_mask: [batch, N_res] mask of non-padded regions.
      pair_act: [batch, N_res, N_res, c_z] pair representation.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_res, c_m].
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    c = self.config
    gc= self.global_config
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert len(pair_act.shape) == 4
    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = hk.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    pair_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='feat_2d_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        init=hk.initializers.RandomNormal(stddev=init_factor))
    bias += jnp.einsum('bqkc,ch->bhqk', pair_act, weights)

    attn_mod = mAttention(
        c, self.global_config, msa_act.shape[-1])
    msa_act,logits = attn_mod(msa_act, msa_act, bias)
    logits=jnp.swapaxes(logits, -2, -3)  #bhqk->bqhk
    logits=jnp.swapaxes(logits, -1, -2)  #bqhk->bqkh
    logits = Linear(
        pair_act.shape[-1],
        initializer='relu',
        name='feat_2d_update_0')(
            logits)
    logits = jax.nn.relu(logits)
    pair_act = Linear(
        pair_act.shape[-1],
        initializer=utils.final_init(gc),
        name='feat_2d_update_1')(
            logits)

    return msa_act,pair_act

class MolEncoder(hk.Module):
  def __init__(self, config, global_config,
               name='mol_encoder'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
              atom_act,
              atom_mask,
              edge_act,
              edge_mask,
              safe_key=None,
              is_training=False):
    """
    Arguments:
        atom_act: [batch, N_atom, c_a] atom representation.
        atom_mask: [batch, N_atom] mask of non-padded regions.
        edge_act: [batch, N_atom, N_atom, c_e] edge representation.
        edge_mask: [batch, N_atom, N_atom] mask of non-padded regions.
        is_training: Whether the module is in training mode.
    Returns:
        Update to atom_act, shape [N_atom, c_a].
        Update to edge_act, shape [N_atom, N_atom, c_e].
    """
    c = self.config
    gc=self.global_config
    assert len(atom_act.shape) == 3
    assert len(atom_mask.shape) == 2
    assert len(edge_act.shape) == 4
    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=self.global_config)
    dropout_wrapper_pair_fn = functools.partial(
        dropout_wrapper_pair,
        is_training=is_training,
        global_config=self.global_config)
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    atom_act = Transition(c['single_transition'], gc, name='atom_transition_input')(
          atom_act,
          atom_mask,
          is_training=is_training, with_norm=False
          )
    edge_act = Transition(c['pair_transition'], gc, name='edge_transition_input')(
          edge_act,
          edge_mask,
          is_training=is_training, with_norm=False
          )

    impl_1=AttentionWithPairBias(c['attention_with_pair_bias'], gc,name='impl_1')
    impl_2=Transition(c['single_transition'], gc, name='impl_2')
    impl_3=Transition(c['pair_transition'], gc, name='impl_3')
    def iteration_fn(x):
      atom_act, edge_act, safe_key = x
      safe_key, *safe_subkey = safe_key.split(4)
      safe_subkey = iter(safe_subkey)
      atom_act,edge_act=dropout_wrapper_pair_fn(
          impl_1,
          atom_act,
          edge_act,
          atom_mask,
          is_training=is_training,
          safe_key=next(safe_subkey))
      atom_act = dropout_wrapper_fn(
          impl_2,
          atom_act,
          atom_mask,
          safe_key=next(safe_subkey))
      edge_act=dropout_wrapper_fn(
          impl_3,
          edge_act,
          edge_mask,
          safe_key=next(safe_subkey))
      return (atom_act, edge_act, safe_key)

    if gc.use_remat:
      iteration_fn = hk.remat(iteration_fn)

    iteration_stack = layer_stack.layer_stack(c['iteration_num_block'])(
        iteration_fn)
    atom_act, edge_act, safe_key = iteration_stack(
        (atom_act, edge_act, safe_key))
    atom_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_atom')(
            atom_act)
    atom_act = Linear(
        c.atom_channels,
        initializer=utils.final_init(gc),
        name='embedding_atom')(
            atom_act)
    edge_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_edge')(
            edge_act)
    edge_act = Linear(
        c.edge_channels,
        initializer=utils.final_init(gc),
        name='embedding_edge')(
            edge_act)

    return atom_act,edge_act

class ProteinEncoder(hk.Module):

  def __init__(self, config, global_config,
               name='protein_encoder'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
              msa_act,
              struc_act,
              msa_mask,
              pair_act,
              pair_mask,
              safe_key=None,
              is_training=False):
    """
    Arguments:
        msa_act: [batch, N_res, c_m] MSA representation.
        msa_mask: [batch, N_res] mask of non-padded regions.
        struc_act: [batch, N_res, c_s] structure representation.
        pair_act: [batch, N_res, N_res, c_p] pair representation.
        pair_mask: [batch, N_res, N_res] mask of non-padded regions.
        is_training: Whether the module is in training mode.
    Returns:
        Update to msa_act, shape [N_res, c_m].
        Update to pair_act, shape [N_res, N_res, c_p].
    """
    c = self.config
    gc=self.global_config
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert len(pair_act.shape) == 4
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    safe_key, *safe_subkey = safe_key.split(5)
    safe_subkey = iter(safe_subkey)
    msa_act = Transition(c['single_transition'], gc, name='single_transition_input')(
          msa_act,
          msa_mask,
          is_training=is_training, with_norm=False
          )
    struc_act =  Transition(c['single_transition'], gc, name='struc_transition_input')(
          struc_act,
          msa_mask,
          is_training=is_training, with_norm=False
          )
    pair_act = Transition(c['pair_transition'], gc, name='pair_transition_input')(
          pair_act,
          pair_mask,
          is_training=is_training, with_norm=False
          )
    struc_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_struc')(
            struc_act)
    msa_act = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='norm_msa')(
            msa_act)
    msa_act=jnp.concatenate((msa_act,struc_act),axis=-1)
    msa_act = Linear(
        c.single_channels*4,
        initializer='relu',
        name='embedding_msa1')(
            msa_act)
    msa_act=jax.nn.relu(msa_act) 
    msa_act = Linear(
        c.single_channels,
        initializer=utils.final_init(gc),
        name='embedding_msa2')(
            msa_act)
    pair_act = Transition(c['pair_transition'], gc, name='pair_transition_input2')(
          pair_act,
          pair_mask,
          is_training=is_training,
          )
    return msa_act,pair_act

class DistMatHead(hk.Module):

  def __init__(self, config, global_config, name='distmat_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair, pair_mask,is_training):
    """
    Arguments:
        pair: [batch, N_res, N_res, c_p] pair representation.
        pair_mask: [batch, N_res] mask of non-padded regions.
        is_training: Whether the module is in training mode.
    Returns:
        logits: [batch, N_res, N_res, 3, num_bins] logits.
        bin_edges: [num_bins] bin edges.
        ret: [batch, N_res, N_res] predicted distance matrix.
    """
    half_logits = Linear(
        self.config.num_bins*3,
        initializer=utils.final_init(self.global_config),
        name='half_logits')(pair)
    logits = half_logits + jnp.swapaxes(half_logits, -2, -3)
    size=logits.shape[-2]
    logits = jnp.reshape(logits,(-1,size,size,3,self.config.num_bins))
    step1=(self.config.middle_break-self.config.first_break)/(int(self.config.num_bins/2) - 1)
    step2=(self.config.last_break-self.config.middle_break)/(int(self.config.num_bins/2) - 2)

    breaks = jnp.concatenate((jnp.linspace(self.config.first_break, self.config.middle_break,
                          int(self.config.num_bins/2)), \
             jnp.linspace(self.config.middle_break + step2, self.config.last_break,
                          int(self.config.num_bins/2) - 1)))
    centers = jnp.concatenate((jnp.linspace(self.config.first_break, self.config.middle_break,
                          int(self.config.num_bins/2)) - step1/2, \
             jnp.linspace(self.config.middle_break + step2, self.config.last_break+ step2,
                          int(self.config.num_bins/2)) - step2/2))
    ret=_calculate_value_from_logits(logits,centers)
    return dict(logits=logits, bin_edges=breaks,ret=ret)

  def loss(self, value, batch):
    logits=value['distmat']['logits']
    bin_edges=jnp.squeeze(value['distmat']['bin_edges'])
    num_bins=self.config.num_bins
    assert len(logits.shape) == 5
    true_dist = jnp.expand_dims(batch['distmat'],axis=-1)
    true_bins = jnp.sum(true_dist > bin_edges, axis=-1)

    errors = softmax_cross_entropy(
        labels=jnp.expand_dims(jax.nn.one_hot(true_bins, num_bins),axis=-2), logits=logits)
    protein_mask = batch['msa_mask']
    atom_mask = batch['atom_mask']
    distmat_mask= batch['distmat_mask']
    protein_mask=jnp.pad(protein_mask,((0,0),(0,distmat_mask.shape[-1]-protein_mask.shape[-1])),'constant',constant_values=(0,0)) * distmat_mask
    atom_mask=jnp.pad(atom_mask,((0,0),(distmat_mask.shape[-1]-atom_mask.shape[-1],0)),'constant',constant_values=(0,0)) * distmat_mask

    square_mask_protein = jnp.expand_dims(protein_mask, axis=-2) * jnp.expand_dims(protein_mask, axis=-1)
    square_mask_atom = jnp.expand_dims(atom_mask, axis=-2) * jnp.expand_dims(atom_mask, axis=-1)
    square_mask_distmat = jnp.expand_dims(distmat_mask, axis=-2) * jnp.expand_dims(distmat_mask, axis=-1) - square_mask_protein - square_mask_atom

    avg_error = (
        jnp.sum(jnp.squeeze(errors[...,0]) * square_mask_protein, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask_protein, axis=(-2, -1))))   

    avg_error += (
        jnp.sum(jnp.squeeze(errors[...,1]) * square_mask_atom, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask_atom, axis=(-2, -1))))
    avg_error += (
        jnp.sum(jnp.squeeze(errors[...,2]) * square_mask_distmat, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask_distmat, axis=(-2, -1))))
    return avg_error
  
class DistMatErrorHead(hk.Module):

  def __init__(self, config, global_config, name='distmat_error_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair, pair_mask,is_training):
    """
    Arguments:
        pair: [batch, N_res, N_res, c_p] pair representation.
        pair_mask: [batch, N_res] mask of non-padded regions.
        is_training: Whether the module is in training mode.
    Returns:
        logits: [batch, N_res, N_res, 3, num_bins] logits.
        bin_edges: [num_bins] bin edges.
        ret: [batch, N_res, N_res] predicted distance matrix.
    """
    half_logits = Linear(
        self.config.num_bins*3,
        initializer=utils.final_init(self.global_config),
        name='half_logits')(pair)
    logits = half_logits + jnp.swapaxes(half_logits, -2, -3)
    size=logits.shape[-2]

    logits = jnp.reshape(logits,(-1,size,size,3,self.config.num_bins))
    step1=(self.config.middle_break-self.config.first_break)/(int(self.config.num_bins/2) - 1)
    step2=(self.config.last_break-self.config.middle_break)/(int(self.config.num_bins/2) - 2)

    breaks = jnp.concatenate((jnp.linspace(self.config.first_break, self.config.middle_break,
                          int(self.config.num_bins/2)), \
             jnp.linspace(self.config.middle_break + step2, self.config.last_break,
                          int(self.config.num_bins/2) - 1)))
    centers = jnp.concatenate((jnp.linspace(self.config.first_break, self.config.middle_break,
                          int(self.config.num_bins/2)) - step1/2, \
             jnp.linspace(self.config.middle_break + step2, self.config.last_break+ step2,
                          int(self.config.num_bins/2)) - step2/2))
    ret=_calculate_value_from_logits(logits,centers)

    return dict(logits=logits, bin_edges=breaks,ret=ret)

  def loss(self, value, batch):
    error_logits=value['distmat_error']['logits']
    pAE=value['distmat_error']['ret']
    assert len(error_logits.shape) == 5
    protein_mask = batch['msa_mask']
    atom_mask = batch['atom_mask']
    distmat_mask= batch['distmat_mask']
    protein_mask=jnp.pad(protein_mask,((0,0),(0,distmat_mask.shape[-1]-protein_mask.shape[-1])),'constant',constant_values=(0,0)) * distmat_mask
    atom_mask=jnp.pad(atom_mask,((0,0),(distmat_mask.shape[-1]-atom_mask.shape[-1],0)),'constant',constant_values=(0,0)) * distmat_mask

    square_mask_protein = jnp.expand_dims(protein_mask, axis=-2) * jnp.expand_dims(protein_mask, axis=-1)
    square_mask_atom = jnp.expand_dims(atom_mask, axis=-2) * jnp.expand_dims(atom_mask, axis=-1)
    square_mask_distmat = jnp.expand_dims(distmat_mask, axis=-2) * jnp.expand_dims(distmat_mask, axis=-1) - square_mask_protein - square_mask_atom

    pMAE_protein=(jnp.sum(jnp.squeeze(pAE[...,0]) * square_mask_protein, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask_protein, axis=(-2, -1)))) 
    pMAE_mol=(jnp.sum(jnp.squeeze(pAE[...,1]) * square_mask_atom, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask_atom, axis=(-2, -1)))) 
    pMAE_distmat=(jnp.sum(jnp.squeeze(pAE[...,2]) * square_mask_distmat, axis=(-2, -1)) /
        (1e-6 + jnp.sum(square_mask_distmat, axis=(-2, -1)))) 
    pMAEs=jnp.concatenate((pMAE_protein[:,None],pMAE_mol[:,None],pMAE_distmat[:,None]),axis=-1)
    assert len(pMAEs.shape)==2
    assert pMAEs.shape[-1]==3
    return pMAEs,pMAEs,pMAEs

class AffinityHead(hk.Module):
  def __init__(self,config, global_config, name='affinity_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
  def __call__(self,single,msa_mask,atom_mask,pair,resi_num,atom_num,safe_key,is_training):
    """
    Arguments:
        single: [batch, N_res, c_p] pair representation.
        msa_mask: [batch, N_res] mask of non-padded regions.
        atom_mask: [batch, N_res] mask of non-padded regions.
        pair: [batch, N_res, N_res, c_p] pair representation.
        resi_num: [batch] number of residues.
        atom_num: [batch] number of atoms.
        is_training: Whether the module is in training mode.
    Returns:
        ret: [batch, 1] predicted affinity
    """
    c=self.config
    gc=self.global_config
    batch_size=single.shape[0]
    num_channels=c.num_channels
    crop_size=msa_mask.shape[-1]
    atom_crop_size=atom_mask.shape[-1]

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper_pair,
        is_training=is_training,
        global_config=gc)

    single = hk.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='input_layer_norm1')(
            single)

    single = Linear(
        num_channels,
        initializer='relu',
        name='act_0')(
            single)
    single = jax.nn.relu(single)
    single = Linear(
        num_channels,
        initializer=utils.final_init(gc),
        name='act_1')(
            single)
            
    protein_act = jnp.einsum('bqc,bq->bqc',single, jnp.pad(msa_mask,((0,0),(0,atom_crop_size)),'constant',constant_values=(0,0)))
    protein_act = jnp.einsum('bqc,b->bc',protein_act,1/resi_num)
    mol_act = jnp.einsum('bqc,bq->bqc',single, jnp.pad(atom_mask,((0,0),(crop_size,0)),'constant',constant_values=(0,0)))
    mol_act = jnp.einsum('bqc,b->bc',mol_act,1/atom_num)

    single=jnp.concatenate((protein_act,mol_act),axis=-1)
    single = Linear(
        num_channels*2,
        initializer='relu',
        name='act_2')(
            single)
    single = jax.nn.relu(single)
    single = Linear(
        num_channels,
        initializer='relu',
        name='act_3')(
            single)
    single = jax.nn.relu(single)
    ret = Linear(
        1,
        initializer=utils.final_init(gc),
        name='output')(
            single)

    return dict(ret=ret)

  def loss(self,value,feat):
    affinity=jnp.expand_dims(feat['affinity'],axis=-1)
    rets=value['affinity']['ret']
    errors = jnp.square(affinity-rets)
    return errors

class LigandTransformer(hk.Module):
  def __init__(self, config, name='LigandTransformer'):
    super().__init__(name=name)
    self.config = config.model
    self.global_config = config.global_config
  def __call__(
      self,
      feat,
      is_training,
      compute_loss=False):
    """
    Arguments:
        feat: dict of input features
        is_training: Whether the module is in training mode.
        compute_loss: Whether to compute loss.
    Returns:
        ret: dict of output features
        loss: loss
    """

    c=self.config
    gc=self.global_config


    msa_act=feat['msa_act']
    struc_act=feat['struc_act']
    msa_mask=feat['msa_mask']
    pair_act=feat['pair_act']
    pair_mask=feat['pair_mask']
    resi_num=feat['resi_num']
    atom_act=feat['atom_act']
    atom_mask=feat['atom_mask']
    edge_act=feat['edge_act']
    edge_mask=feat['edge_mask']
    resi_num=feat['resi_num']
    atom_num=feat['atom_num']
    distmat_mask=feat['distmat_mask']
    crop_size=msa_act.shape[-2]
    atom_crop_size=atom_act.shape[-2]
    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=self.global_config)
    dropout_wrapper_pair_fn = functools.partial(
        dropout_wrapper_pair,
        is_training=is_training,
        global_config=self.global_config)
    safe_key = prng.SafeKey(hk.next_rng_key())
    safe_key, *sub_keys = safe_key.split(3)
    sub_keys = iter(sub_keys)

    msa_act,pair_act=ProteinEncoder(c.protein_encoder,gc)(msa_act,struc_act,msa_mask,pair_act,pair_mask,
      safe_key=next(sub_keys),is_training=is_training)
    atom_act,edge_act=MolEncoder(c.mol_encoder,gc)(atom_act,atom_mask,edge_act,edge_mask,
      safe_key=next(sub_keys),is_training=is_training)

    single=jnp.concatenate((msa_act,atom_act),axis=-2)
    pair_act=jnp.pad(pair_act,((0,0),(0,0),(0,atom_crop_size),(0,0)),'constant',constant_values=(0.0,0.0))
    edge_act=jnp.pad(edge_act,((0,0),(0,0),(crop_size,0),(0,0)),'constant',constant_values=(0.0,0.0))
    pair=jnp.concatenate((pair_act,edge_act),axis=-3)
    del pair_act,edge_act

    single_wise_mask=jnp.concatenate((msa_mask,atom_mask),axis=-1)

    pair_wise_mask=jnp.concatenate(
                        (jnp.concatenate((pair_mask,jnp.einsum('bi,bj->bij',msa_mask,atom_mask)),axis=-1),
                         jnp.concatenate((jnp.einsum('bi,bj->bij',atom_mask,msa_mask),edge_mask),axis=-1)),
                         axis=-2)

    impl_1=AttentionWithPairBias(c['attention_with_pair_bias'], gc,name='impl_1')
    impl_2=Transition(c['single_transition'], gc, name='impl_2')
    impl_3=Transition(c['pair_transition'], gc, name='impl_3')
    def iteration_fn(x):
      single, pair, safe_key = x
      safe_key, *safe_subkey = safe_key.split(4)
      safe_subkey = iter(safe_subkey)
      single,pair=dropout_wrapper_pair_fn(
          impl_1,
          single,
          pair,
          single_wise_mask,
          is_training=is_training,
          safe_key=next(safe_subkey))
      single = dropout_wrapper_fn(
          impl_2,
          single,
          single_wise_mask,
          safe_key=next(safe_subkey))
      pair=dropout_wrapper_fn(
          impl_3,
          pair,
          pair_wise_mask,
          safe_key=next(safe_subkey))
      return (single, pair, safe_key)
      
    if gc.use_remat:
      iteration_fn = hk.remat(iteration_fn)

    iteration_stack = layer_stack.layer_stack(c['iteration_num_block'])(
        iteration_fn)
    single, pair, safe_key = iteration_stack(
        (single, pair, safe_key))

    ret={}
    affinity_head=AffinityHead(c['affinity_head'],gc,name='affinity_head')
    distmat_head=DistMatHead(c['distmat_head'],gc,name='distmat_head')
    distmat_error_head=DistMatErrorHead(c['distmat_error_head'],gc,name='distmat_error_head')
    ret['affinity']=affinity_head(single,msa_mask,atom_mask,pair,resi_num,atom_num,safe_key,is_training)
    ret['distmat']=distmat_head(pair,distmat_mask,is_training)
    ret['distmat_error']=distmat_error_head(pair,distmat_mask,is_training)
    if not compute_loss:
      return ret
    else:
      loss=affinity_head.loss(ret,feat)*affinity_head.config.weight
      _,pMAEs,MAEs=distmat_error_head.loss(ret,feat)
      ret['distmat_error']['pMAEs']=pMAEs
      ret['distmat_error']['MAEs']=MAEs
      return ret,loss