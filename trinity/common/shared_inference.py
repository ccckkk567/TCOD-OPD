"""Opt-in two-GPU layout: shared inference GPU plus dedicated trainer GPU."""

import math


def validate_shared_inference(config):
    if config.mode != "both" or (config.cluster.node_num, config.cluster.gpu_per_node) != (1, 2):
        raise ValueError("Shared inference requires mode=both on one node with two GPUs")
    if len(config.explorer.auxiliary_models) != 1:
        raise ValueError("Shared inference requires exactly one teacher")
    models = [config.explorer.rollout_model, config.explorer.auxiliary_models[0]]
    for model in models:
        if model.engine_num != 1 or model.tensor_parallel_size != 1:
            raise ValueError("Shared inference requires one TP=1 engine per model")
        if not model.engine_type.startswith("vllm") or not model.enforce_eager:
            raise ValueError("Shared inference requires eager vLLM engines")
        if (
            not math.isfinite(model.gpu_memory_utilization)
            or not 0 < model.gpu_memory_utilization < 1
        ):
            raise ValueError("Invalid shared inference memory budget")
    if sum(model.gpu_memory_utilization for model in models) > 0.85:
        raise ValueError("Shared inference budgets must leave at least 15% device headroom")
    if config.trainer.ulysses_sequence_parallel_size != 1:
        raise ValueError("The single training GPU requires sequence parallel size 1")
    if config.algorithm.algorithm_type != "on_policy_distill":
        raise ValueError("Shared inference is currently approved only for OPD")


def create_shared_inference(config, engine_cls):
    """Reserve ONE physical GPU and initialize two fractional actors sequentially."""
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    validate_shared_inference(config)
    pg = placement_group([{"GPU": 1}], strategy="PACK")
    ray.get(pg.ready(), timeout=600)
    actors, allocations = [], []
    for role, model in (
        ("rollout_model_0", config.explorer.rollout_model),
        ("auxiliary_model_0_0", config.explorer.auxiliary_models[0]),
    ):
        model.ray_namespace = config.ray_namespace
        model.bundle_indices = "0"
        if role.startswith("auxiliary"):
            model.enable_openai_api = True
            model.engine_type = "vllm"
        actor = (
            ray.remote(engine_cls)
            .options(
                name=f"{config.explorer.name}_{role}",
                namespace=config.ray_namespace,
                num_cpus=0,
                num_gpus=0.5,
                # EngineCore is a spawned non-Ray process. Its first ray.get_actor
                # must join this job, not auto-start another local cluster.
                runtime_env={"env_vars": {"RAY_ADDRESS": ray.get_runtime_context().gcs_address}},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                ),
            )
            .remote(config=model)
        )
        allocation = ray.get(actor.shared_device_info.remote(), timeout=900)
        if len(allocation["gpu_ids"]) != 1 or "H200" not in allocation["name"]:
            raise RuntimeError(f"Expected one H200 per shared inference actor: {allocation}")
        if allocations and allocation["gpu_ids"] != allocations[0]["gpu_ids"]:
            raise RuntimeError("Inference actors were not placed on the same physical GPU")
        print(f"SHARED_INFERENCE {role}: {allocation}", flush=True)
        actors.append(actor)
        allocations.append(allocation)
    return [actors[0]], [[actors[1]]]
