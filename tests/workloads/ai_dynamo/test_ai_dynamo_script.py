# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "src/cloudai/workloads/ai_dynamo/ai_dynamo.sh"


def run_script_functions(body: str) -> list[str]:
    command = f"source {shlex.quote(str(SCRIPT))}; {body}"
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()


def common_role_setup(backend: str, current_node: str) -> str:
    return f"""
        dynamo_args["backend"]={backend}
        dynamo_args["prefill-port"]=30011
        prefill_config["num-nodes"]=2
        prefill_config["nodes-per-worker"]=2
        prefill_config["node-list"]=n0,n1
        prefill_config["multiple-workers-per-node"]=False
        prefill_args["--tensor-parallel-size"]=16
        prefill_args["--pipeline-parallel-size"]=1
        prefill_args["--data-parallel-size"]=1
        decode_config["num-nodes"]=1
        decode_config["nodes-per-worker"]=1
        decode_config["multiple-workers-per-node"]=False
        decode_args["--tensor-parallel-size"]=1
        decode_args["--pipeline-parallel-size"]=1
        decode_args["--data-parallel-size"]=1
        SLURMD_NODENAME={current_node}
        _resolve_host_ip() {{ echo "10.0.0.1"; }}
    """


def test_vllm_multinode_worker_injects_mp_rank_arguments() -> None:
    lines = run_script_functions(
        common_role_setup("vllm", "n1")
        + """
        _validate_worker_topology
        _apply_multinode_role_args prefill
        printf '%s|%s|%s|%s|%s|%s|%s\n' \
          "${prefill_args["--distributed-executor-backend"]}" \
          "${prefill_args["--nnodes"]}" \
          "${prefill_args["--node-rank"]}" \
          "${prefill_args["--master-addr"]}" \
          "${prefill_args["--master-port"]}" \
          "$(_role_group_leader prefill)" \
          "$(_role_is_vllm_headless prefill && echo true || echo false)"
        """
    )

    assert lines[-1] == "mp|2|1|10.0.0.1|30011|n0|true"


def test_sglang_multinode_worker_injects_native_rank_arguments() -> None:
    lines = run_script_functions(
        common_role_setup("sglang", "n0")
        + """
        _validate_worker_topology
        _apply_multinode_role_args prefill
        printf '%s|%s|%s|%s\n' \
          "${prefill_args["--dist-init-addr"]}" \
          "${prefill_args["--nnodes"]}" \
          "${prefill_args["--node-rank"]}" \
          "$(_role_is_group_leader prefill && echo true || echo false)"
        """
    )

    assert lines[-1] == "10.0.0.1:30011|2|0|true"


def test_multinode_allocation_uses_local_world_size_and_one_process_per_node() -> None:
    lines = run_script_functions(
        """
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        SHARED_NODE_DISAGG=false
        prefill_config["nodes-per-worker"]=2
        prefill_config["multiple-workers-per-node"]=False
        decode_config["nodes-per-worker"]=2
        decode_config["multiple-workers-per-node"]=False
        prefill_args["--tensor-parallel-size"]=16
        prefill_args["--pipeline-parallel-size"]=1
        decode_args["--tensor-parallel-size"]=16
        decode_args["--pipeline-parallel-size"]=1
        _compute_worker_allocation_vllm
        printf '%s|%s|%s|%s\n' \
          "${prefill_config["gpus-per-worker"]}" \
          "${prefill_config["workers-per-node"]}" \
          "${decode_config["gpus-per-worker"]}" \
          "${decode_config["workers-per-node"]}"
        """
    )

    assert lines[-1] == "8|1|8|1"


def test_ready_count_tracks_logical_workers_not_participating_nodes() -> None:
    lines = run_script_functions(
        """
        prefill_config["num-nodes"]=4
        prefill_config["nodes-per-worker"]=2
        prefill_config["workers-per-node"]=1
        decode_config["num-nodes"]=2
        decode_config["nodes-per-worker"]=2
        decode_config["workers-per-node"]=1
        printf '%s|%s\n' "$(_expected_ready_prefill)" "$(_expected_ready_decode)"
        """
    )

    assert lines[-1] == "2|1"


def test_omitted_nodes_per_worker_preserves_one_worker_per_node() -> None:
    lines = run_script_functions(
        """
        dynamo_args["backend"]=vllm
        prefill_config["num-nodes"]=3
        prefill_config["workers-per-node"]=1
        decode_config["num-nodes"]=2
        decode_config["workers-per-node"]=1
        _normalize_worker_topology
        printf '%s|%s|%s|%s\n' \
          "${prefill_config["nodes-per-worker"]}" \
          "$(_total_workers_prefill)" \
          "${decode_config["nodes-per-worker"]}" \
          "$(_total_workers_decode)"
        """
    )

    assert lines[-1] == "1|3|1|2"


def test_runtime_selects_dynamo_131_compatible_communication_planes() -> None:
    lines = run_script_functions(
        """
        dynamo_args["backend"]=test
        dynamo_args["frontend-node"]=n0
        dynamo_args["nats-port"]=4222
        dynamo_args["etcd-port"]=2379
        RESULTS_DIR=/tmp
        _init_runtime_env
        printf '%s|%s\n' "$DYN_DISCOVERY_BACKEND" "$DYN_EVENT_PLANE"
        """
    )

    assert lines[-1] == "etcd|nats"
