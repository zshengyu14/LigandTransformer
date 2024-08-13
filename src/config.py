import ml_collections
import copy

def model_config() -> ml_collections.ConfigDict:
  cfg = copy.deepcopy(CONFIG)
  return cfg

CONFIG = ml_collections.ConfigDict({
                'global_config': {
                    'deterministic': False,
                    'subbatch_size': 4,
                    'use_remat': True,
                    'zero_init': False
                },
                'model':{
                    'attention_with_pair_bias': {
                        'dropout_rate': 0.15,
                        'gating': True,
                        'num_head': 8,
                        'shared_dropout': True
                    },
                    'single_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 4,
                        'shared_dropout': True
                    },
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'iteration_num_block':12
                    ,
                    'mol_encoder': {
                        'atom_channels':384,
                        'edge_channels':128,
                        'iteration_num_block':4,
                        'single_transition': {
                            'dropout_rate': 0.15,
                            'num_intermediate_factor': 4,
                            'shared_dropout': True
                        },
                        'pair_transition': {
                            'dropout_rate': 0.0,
                            'num_intermediate_factor': 4,
                            'orientation': 'per_row',
                            'shared_dropout': True
                        },
                        'attention_with_pair_bias': {
                            'dropout_rate': 0.15,
                            'gating': True,
                            'num_head': 6,
                            'shared_dropout': True
                        },
                    },
                    'protein_encoder': {
                        'single_channels':384,
                        'pair_channels':128,
                        'single_transition': {
                            'dropout_rate': 0.0,
                            'num_intermediate_factor': 4,
                            'shared_dropout': True
                        },
                        'pair_transition': {
                            'dropout_rate': 0.0,
                            'num_intermediate_factor': 4,
                            'orientation': 'per_row',
                            'shared_dropout': True
                        },
                    },
                    'affinity_head': {
                        'num_bins': 50,
                        'num_channels':1024,
                        'weight': 1.0
                    },
                    'distmat_head': {
                        'first_break': 1.0,
                        'middle_break': 10.5,
                        'last_break': 20.0,
                        'num_bins': 100,
                        'weight': 50.0
                    },
                },
                'data':{
                    'pair_channels':128,
                    'single_channels':384,
                    'atom_embeddings_channels':300,
                    'edge_embeddings_channels':300,
                }

})