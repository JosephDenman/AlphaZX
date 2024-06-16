from easydict import EasyDict

from alphazx.diagram import NODE_METADATA, POSSIBLE_PHASES

alphazx_ppo_config = dict(
    max_num_qubits=50,
    max_circuit_depth=50,
    t_gates=True,
    max_num_new_edges=10,
    num_phase_buckets=10,
    done_reward=1.,
    step_penalty=-1.,
    max_num_steps=100,
    exp_name='alphazx_ppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        model=dict(
            num_node_types=len(NODE_METADATA),
            num_possible_phases=len(POSSIBLE_PHASES),
            num_possible_new_edges=10,
            node_embedding_channels=(1 + len(POSSIBLE_PHASES) + 10),
            repr_gps_channels=(1 + len(POSSIBLE_PHASES) + 10 + 2),
            repr_gps_edge_in_channels=2,
            repr_gps_edge_out_channels=2,
            repr_gps_pe_in_channels=2,
            repr_gps_pe_out_channels=2,
            repr_gps_num_layers=5,
            repr_gps_bias=True,
            repr_gps_num_attn_heads=1,
            repr_gps_attn_type='multihead',
            repr_gps_attn_kwargs={},
            repr_gps_mlp_hidden_channels=64,
            policy_num_pooling_encoder_blocks=4,
            policy_num_pooling_heads=3,
            policy_pooling_layer_norm=True,
            policy_pooling_dropout=0.1,
            value_gmt_num_encoder_blocks=4,
            value_gmt_num_heads=3,
            value_gmt_layer_norm=True,
            value_gmt_dropout=0.1
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
alphazx_ppo_config = EasyDict(alphazx_ppo_config)
main_config = alphazx_ppo_config
alphazx_ppo_create_config = dict(
    env=dict(
        type='alphazx',
        import_names=['alphazx.ppo.env.alphazx_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='alphazx_ppo'),
)
alphazx_ppo_create_config = EasyDict(alphazx_ppo_create_config)
create_config = alphazx_ppo_create_config

