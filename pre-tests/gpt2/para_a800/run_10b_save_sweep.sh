#!/bin/bash

set -euo pipefail

WORKDIR="/workspace/Megatron-LM"
REMOTE_HELPER="$WORKDIR/pre-tests/gpt2/para_a800/run_remote_parallel.sh"
MODEL_SIZE="${MODEL_SIZE:-10B}"
case "$MODEL_SIZE" in
    2.7B) MODEL_DIR=2.7B-4nodes; MODEL_SLUG=2.7b ;;
    7B) MODEL_DIR=7B-4nodes; MODEL_SLUG=7b ;;
    10B) MODEL_DIR=10B-4nodes; MODEL_SLUG=10b ;;
    14B) MODEL_DIR=14B-4nodes; MODEL_SLUG=14b ;;
    20B) MODEL_DIR=20B-4nodes; MODEL_SLUG=20b ;;
    *) echo "MODEL_SIZE must be 2.7B, 7B, 10B, 14B, or 20B." >&2; exit 2 ;;
esac
SCRIPT_DIR="pre-tests/gpt2/para_a800/$MODEL_DIR"
CHECKPOINT_PREFIX="/dev/shm/models/gpt2-$MODEL_SLUG-4nodes"
SWEEP_TAG="${MODEL_SLUG}_save_sweep"
HOSTS=(node1 node2 node3)
ALL_NODES=(node0 node1 node2 node3)
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
RDMA_HCA_PROFILE="${RDMA_HCA_PROFILE:-full}"
ECCHECK_DATA_BUFFERS_COUNT="${ECCHECK_DATA_BUFFERS_COUNT:-12}"
GEMINI_ENV_VARS=(
    GEMINI_GDR
    GEMINI_GDR_MIRROR_MODE
    GEMINI_GDR_BATCH_WR
    GEMINI_MIRROR_CHUNK_MB
    GEMINI_REPLICAS_CHANNELS_PER_PEER
    GEMINI_SAVE_PREFER_TORCH_PINNED
    GEMINI_PIPELINE_CHUNK_MB
    GEMINI_PIPELINE_MAX_SEGMENTS
    GEMINI_PIPELINE_SEGMENTS
)

build_gemini_env_prefix() {
    local name value prefix=""
    printf -v value '%q' "${GEMINI_GDR:-1}"
    prefix="GEMINI_GDR=$value"
    for name in "${GEMINI_ENV_VARS[@]:1}"; do
        if [[ -v $name ]]; then
            printf -v value '%q' "${!name}"
            prefix+=" $name=$value"
        fi
    done
    printf '%s' "$prefix"
}

DEFAULT_SCHEMES=(gemini2 gemini3 frcheck eccheck ecnaive)
declare -A SCRIPTS=(
    [gemini2]="$SCRIPT_DIR/test_gemini_2.sh"
    [gemini3]="$SCRIPT_DIR/test_gemini_3.sh"
    [frcheck]="$SCRIPT_DIR/test_frcheck.sh"
    [eccheck]="$SCRIPT_DIR/test_eccheck.sh"
    [ecnaive]="$SCRIPT_DIR/test_ecnaive.sh"
)
declare -A CHECKPOINT_PATHS=(
    [gemini2]="$CHECKPOINT_PREFIX-gemini-2-replicas"
    [gemini3]="$CHECKPOINT_PREFIX-gemini-3-replicas"
    [frcheck]="$CHECKPOINT_PREFIX-frcheck"
    [eccheck]="$CHECKPOINT_PREFIX-eccheck"
    [ecnaive]="$CHECKPOINT_PREFIX-ecnaive"
)
declare -A MASTER_PORTS=(
    [gemini2]="${MASTER_PORT_GEMINI2:-6100}"
    [gemini3]="${MASTER_PORT_GEMINI3:-6110}"
    [frcheck]="${MASTER_PORT_FRCHECK:-6120}"
    [eccheck]="${MASTER_PORT_ECCHECK:-6130}"
    [ecnaive]="${MASTER_PORT_ECNAIVE:-6140}"
)

usage() {
    echo "Usage: $0 [--dry-run] [gemini2|gemini3|frcheck|eccheck|ecnaive ...]"
    echo "Environment: MODEL_SIZE=2.7B|7B|10B|14B|20B, LOG_DIR, CONTINUE_ON_ERROR=0|1, DRY_RUN=0|1"
    echo "             RDMA_HCA_PROFILE=full|half|quarter (default: full)"
    echo "             ECCHECK_DATA_BUFFERS_COUNT (default: 12; positive integer)"
    echo "             SSH_USER, SSH_PORT, SSH_CONNECT_TIMEOUT, SSH_IDENTITY_FILE"
    echo "             MASTER_PORT_GEMINI2, MASTER_PORT_GEMINI3, MASTER_PORT_FRCHECK,"
    echo "             MASTER_PORT_ECCHECK, MASTER_PORT_ECNAIVE"
    echo "             FRCHECK_LAYER_FRONTIER_ORDER (default: chunk_layer_sid)"
    echo "             FRCHECK_SAVE_PREFER_TORCH_PINNED=0|1 (default: 0)"
    echo "             FRCHECK_LAYER_STREAM_ENCODE (default: 1), FRCHECK_LAYER_ENCODE_COALESCE_US (default: 0)"
    echo "             FRCHECK_SEND_LANES_PER_PEER (default: 12), FRCHECK_RECV_LANES_PER_PEER (default: 12)"
    echo "             FRCHECK_NET_PHYSICAL_CORES (default: 0), FRCHECK_RDMA_CHUNK_MB (default: 64)"
    echo "             GEMINI_GDR, GEMINI_GDR_MIRROR_MODE, GEMINI_GDR_BATCH_WR,"
    echo "             GEMINI_MIRROR_CHUNK_MB, GEMINI_REPLICAS_CHANNELS_PER_PEER,"
    echo "             GEMINI_SAVE_PREFER_TORCH_PINNED, GEMINI_PIPELINE_CHUNK_MB,"
    echo "             GEMINI_PIPELINE_MAX_SEGMENTS, GEMINI_PIPELINE_SEGMENTS"
}

schemes=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        gemini2|gemini3|frcheck|eccheck|ecnaive) schemes+=("$argument") ;;
        *) echo "Unknown scheme or option: $argument" >&2; usage >&2; exit 2 ;;
    esac
done
[[ ${#schemes[@]} -gt 0 ]] || schemes=("${DEFAULT_SCHEMES[@]}")
[[ "$CONTINUE_ON_ERROR" == 0 || "$CONTINUE_ON_ERROR" == 1 ]] || {
    echo "CONTINUE_ON_ERROR must be 0 or 1." >&2
    exit 2
}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
}
[[ "$RDMA_HCA_PROFILE" == full || "$RDMA_HCA_PROFILE" == half || "$RDMA_HCA_PROFILE" == quarter ]] || {
    echo "RDMA_HCA_PROFILE must be full, half, or quarter." >&2
    exit 2
}
for scheme in "${schemes[@]}"; do
    if [[ "$scheme" == eccheck && ! "$ECCHECK_DATA_BUFFERS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        echo "ECCHECK_DATA_BUFFERS_COUNT must be a positive integer." >&2
        exit 2
    fi
done

SSH_OPTIONS=(-n -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" -o StrictHostKeyChecking=accept-new)
[[ -z "$SSH_IDENTITY_FILE" ]] || SSH_OPTIONS+=(-i "$SSH_IDENTITY_FILE")

ssh_run() {
    local host=$1 command=$2 quoted_command
    printf -v quoted_command '%q' "$command"
    ssh "${SSH_OPTIONS[@]}" "${SSH_USER}@${host}" "bash -lc $quoted_command"
}

validate_checkpoint_path() {
    local checkpoint_path=$1 scheme
    [[ -n "$checkpoint_path" && "$checkpoint_path" != "/" ]] || return 1
    for scheme in "${DEFAULT_SCHEMES[@]}"; do
        [[ "$checkpoint_path" == "${CHECKPOINT_PATHS[$scheme]}" ]] && return 0
    done
    return 1
}

check_no_training_processes() {
    local node=$1 output="" command="pgrep -af '[t]orchrun|pretrain_gpt[.]py'"
    if [[ "$node" == node0 ]]; then
        output=$(bash -lc "$command" || true)
    else
        output=$(ssh_run "$node" "$command" || true)
    fi
    if [[ -n "$output" ]]; then
        echo "Existing training process found on $node:" >&2
        echo "$output" >&2
        return 1
    fi
}

preflight() {
    local scheme host script checkpoint_path
    [[ -d "$WORKDIR" ]] || { echo "Work directory not found: $WORKDIR" >&2; return 1; }
    [[ -x "$REMOTE_HELPER" ]] || { echo "Remote helper is not executable: $REMOTE_HELPER" >&2; return 1; }
    if [[ -n "$SSH_IDENTITY_FILE" && ! -f "$SSH_IDENTITY_FILE" ]]; then
        echo "SSH identity file not found: $SSH_IDENTITY_FILE" >&2
        return 1
    fi
    for scheme in "${schemes[@]}"; do
        script=${SCRIPTS[$scheme]}
        checkpoint_path=${CHECKPOINT_PATHS[$scheme]}
        [[ -f "$WORKDIR/$script" ]] || { echo "Training script not found: $WORKDIR/$script" >&2; return 1; }
        validate_checkpoint_path "$checkpoint_path" || { echo "Unsafe checkpoint path for $scheme: $checkpoint_path" >&2; return 1; }
    done
    [[ "$DRY_RUN" == 0 ]] || return 0
    check_no_training_processes node0 || return 1
    for host in "${HOSTS[@]}"; do
        ssh_run "$host" "true" >/dev/null || { echo "SSH check failed for $host" >&2; return 1; }
        check_no_training_processes "$host" || return 1
        for scheme in "${schemes[@]}"; do
            script=${SCRIPTS[$scheme]}
            ssh_run "$host" "test -d '$WORKDIR' && test -f '$WORKDIR/$script'" || {
                echo "Remote path check failed on $host for $WORKDIR/$script" >&2
                return 1
            }
        done
    done
}

preflight || exit 1
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry-run command plan:"
    for scheme in "${schemes[@]}"; do
        script=${SCRIPTS[$scheme]}
        checkpoint_path=${CHECKPOINT_PATHS[$scheme]}
        port=${MASTER_PORTS[$scheme]}
        echo "SCHEME=$scheme PORT=$port PATH=$checkpoint_path RDMA_HCA_PROFILE=$RDMA_HCA_PROFILE"
        gemini_env=""
        frcheck_env=""
        eccheck_env=""
        if [[ "$scheme" == frcheck ]]; then
            frcheck_env="FRCHECK_SAVE_PREFER_TORCH_PINNED=${FRCHECK_SAVE_PREFER_TORCH_PINNED:-0} FRCHECK_LAYER_STREAM_ENCODE=${FRCHECK_LAYER_STREAM_ENCODE:-1} FRCHECK_LAYER_ENCODE_COALESCE_US=${FRCHECK_LAYER_ENCODE_COALESCE_US:-0} FRCHECK_SEND_LANES_PER_PEER=${FRCHECK_SEND_LANES_PER_PEER:-12} FRCHECK_RECV_LANES_PER_PEER=${FRCHECK_RECV_LANES_PER_PEER:-12} FRCHECK_NET_PHYSICAL_CORES=${FRCHECK_NET_PHYSICAL_CORES:-0} FRCHECK_RDMA_CHUNK_MB=${FRCHECK_RDMA_CHUNK_MB:-64} "
            echo "  frcheck_config: ${frcheck_env% }"
        fi
        if [[ "$scheme" == gemini2 || "$scheme" == gemini3 ]]; then
            gemini_env="$(build_gemini_env_prefix) "
            echo "  gemini_config: ${gemini_env% }"
        fi
        if [[ "$scheme" == eccheck ]]; then
            eccheck_env="ECCHECK_DATA_BUFFERS_COUNT=$ECCHECK_DATA_BUFFERS_COUNT "
            echo "  eccheck_config: ${eccheck_env% }"
        fi
        echo "  remote: RDMA_HCA_PROFILE=$RDMA_HCA_PROFILE ${frcheck_env}${gemini_env}${eccheck_env}MASTER_PORT=$port ./$script {R} save --train-iters 10"
        echo "  local:  RDMA_HCA_PROFILE=$RDMA_HCA_PROFILE ${frcheck_env}${gemini_env}${eccheck_env}MASTER_PORT=$port ./$script 0 save --train-iters 10"
        echo "  cleanup: rm -rf -- $checkpoint_path on node0,node1,node2,node3"
    done
    exit 0
fi

UTC_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="${LOG_DIR:-$WORKDIR/logs/${SWEEP_TAG}/$UTC_TIMESTAMP}"
python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)' "$LOG_DIR"
[[ -d "$LOG_DIR" ]] || {
    echo "Failed to create log directory: $LOG_DIR" >&2
    exit 1
}
SUMMARY_LOG="$LOG_DIR/summary.log"
: >"$SUMMARY_LOG"
summary() { printf '%s\n' "$*" | tee -a "$SUMMARY_LOG"; }

record_shm_available() {
    local node command output
    command="python3 -c 'import os; s=os.statvfs(\"/dev/shm\"); print(s.f_bavail*s.f_frsize)'"
    for node in "${ALL_NODES[@]}"; do
        if [[ "$node" == node0 ]]; then output=$(bash -lc "$command"); else output=$(ssh_run "$node" "$command"); fi
        summary "SHM_AVAILABLE,node=$node,bytes=$output,gib=$(python3 -c "print(f'{$output / 1073741824:.6f}')")"
    done
}

cleanup_scheme() {
    local scheme=$1 checkpoint_path=${CHECKPOINT_PATHS[$1]} host failed=0
    validate_checkpoint_path "$checkpoint_path" || return 1
    rm -rf -- "$checkpoint_path" || failed=1
    for host in "${HOSTS[@]}"; do ssh_run "$host" "rm -rf -- '$checkpoint_path'" || failed=1; done
    if [[ -e "$checkpoint_path" ]]; then
        summary "CLEANUP,scheme=$scheme,node=node0,status=failed,path=$checkpoint_path"; failed=1
    else
        summary "CLEANUP,scheme=$scheme,node=node0,status=clean,path=$checkpoint_path"
    fi
    for host in "${HOSTS[@]}"; do
        if ssh_run "$host" "test ! -e '$checkpoint_path'"; then
            summary "CLEANUP,scheme=$scheme,node=$host,status=clean,path=$checkpoint_path"
        else
            summary "CLEANUP,scheme=$scheme,node=$host,status=failed,path=$checkpoint_path"; failed=1
        fi
    done
    [[ "$failed" == 0 ]]
}

terminate_jobs() {
    local job_pids
    job_pids=$(jobs -pr) || true
    [[ -z "$job_pids" ]] || kill $job_pids 2>/dev/null || true
}
trap terminate_jobs INT TERM

run_training() {
    local scheme=$1 script=${SCRIPTS[$1]} port=${MASTER_PORTS[$1]}
    local scheme_log="$LOG_DIR/$scheme.log"
    local frcheck_env="" gemini_env="" eccheck_env="" name
    local -a gemini_env_args=() eccheck_env_args=()
    echo "[driver] RDMA_HCA_PROFILE=$RDMA_HCA_PROFILE"
    if [[ "$scheme" == frcheck ]]; then
        echo "[driver] FRCHECK_LAYER_FRONTIER_ORDER=${FRCHECK_LAYER_FRONTIER_ORDER:-chunk_layer_sid}"
        frcheck_env=" FRCHECK_LAYER_EXCHANGE_SEG=${FRCHECK_LAYER_EXCHANGE_SEG:-12} FRCHECK_LAYER_EXCHANGE_CHUNK_MB=${FRCHECK_LAYER_EXCHANGE_CHUNK_MB:-32} FRCHECK_LAYER_FRONTIER_ORDER=${FRCHECK_LAYER_FRONTIER_ORDER:-chunk_layer_sid} FRCHECK_LAYER_ENCODE_BATCH=${FRCHECK_LAYER_ENCODE_BATCH:-24} FRCHECK_LAYER_ENCODE_SUBMIT_WORKER=${FRCHECK_LAYER_ENCODE_SUBMIT_WORKER:-0} FRCHECK_LAYER_ENCODE_ADAPTIVE=${FRCHECK_LAYER_ENCODE_ADAPTIVE:-0} FRCHECK_LAYER_ENCODE_ADAPTIVE_MULTIPLIER=${FRCHECK_LAYER_ENCODE_ADAPTIVE_MULTIPLIER:-3} FRCHECK_LAYER_STREAM_ENCODE=${FRCHECK_LAYER_STREAM_ENCODE:-1} FRCHECK_LAYER_ENCODE_COALESCE_US=${FRCHECK_LAYER_ENCODE_COALESCE_US:-0} FRCHECK_SEND_LANES_PER_PEER=${FRCHECK_SEND_LANES_PER_PEER:-12} FRCHECK_RECV_LANES_PER_PEER=${FRCHECK_RECV_LANES_PER_PEER:-12} FRCHECK_NET_PHYSICAL_CORES=${FRCHECK_NET_PHYSICAL_CORES:-0} FRCHECK_RDMA_CHUNK_MB=${FRCHECK_RDMA_CHUNK_MB:-64} FRCHECK_TRACE_INIT=${FRCHECK_TRACE_INIT:-0} FRCHECK_GDR=${FRCHECK_GDR:-0} FRCHECK_ASYNC_PARITY=${FRCHECK_ASYNC_PARITY:-1} FRCHECK_SAVE_PREFER_TORCH_PINNED=${FRCHECK_SAVE_PREFER_TORCH_PINNED:-0}"
    fi
    if [[ "$scheme" == gemini2 || "$scheme" == gemini3 ]]; then
        gemini_env=" $(build_gemini_env_prefix)"
        gemini_env_args+=("GEMINI_GDR=${GEMINI_GDR:-1}")
        for name in "${GEMINI_ENV_VARS[@]:1}"; do
            [[ ! -v $name ]] || gemini_env_args+=("$name=${!name}")
        done
        echo "[driver] Gemini config:${gemini_env}"
    fi
    if [[ "$scheme" == eccheck ]]; then
        eccheck_env=" ECCHECK_DATA_BUFFERS_COUNT=$ECCHECK_DATA_BUFFERS_COUNT"
        eccheck_env_args+=("ECCHECK_DATA_BUFFERS_COUNT=$ECCHECK_DATA_BUFFERS_COUNT")
        echo "[driver] ECCHECK config:${eccheck_env}"
    fi
    local remote_command="export PRINT_CMD=0 MASTER_PORT=$port RDMA_HCA_PROFILE=$RDMA_HCA_PROFILE$frcheck_env$gemini_env$eccheck_env; ./$script {R} save --train-iters 10"
    (
        local remote_pid local_pid remote_rc local_rc
        trap 'kill "${remote_pid:-}" "${local_pid:-}" 2>/dev/null || true' INT TERM
        SSH_USER="$SSH_USER" SSH_PORT="$SSH_PORT" SSH_CONNECT_TIMEOUT="$SSH_CONNECT_TIMEOUT" \
            SSH_IDENTITY_FILE="$SSH_IDENTITY_FILE" REMOTE_WORKDIR="$WORKDIR" \
            "$REMOTE_HELPER" -r -c "$remote_command" &
        remote_pid=$!
        (
            set -o pipefail
            cd "$WORKDIR"
            env PRINT_CMD=0 MASTER_PORT="$port" RDMA_HCA_PROFILE="$RDMA_HCA_PROFILE" \
                "${gemini_env_args[@]}" \
                "${eccheck_env_args[@]}" \
                FRCHECK_LAYER_EXCHANGE_SEG="${FRCHECK_LAYER_EXCHANGE_SEG:-12}" \
                FRCHECK_LAYER_EXCHANGE_CHUNK_MB="${FRCHECK_LAYER_EXCHANGE_CHUNK_MB:-32}" \
                FRCHECK_LAYER_FRONTIER_ORDER="${FRCHECK_LAYER_FRONTIER_ORDER:-chunk_layer_sid}" \
                FRCHECK_LAYER_ENCODE_BATCH="${FRCHECK_LAYER_ENCODE_BATCH:-24}" \
                FRCHECK_LAYER_ENCODE_SUBMIT_WORKER="${FRCHECK_LAYER_ENCODE_SUBMIT_WORKER:-0}" \
                FRCHECK_LAYER_ENCODE_ADAPTIVE="${FRCHECK_LAYER_ENCODE_ADAPTIVE:-0}" \
                FRCHECK_LAYER_ENCODE_ADAPTIVE_MULTIPLIER="${FRCHECK_LAYER_ENCODE_ADAPTIVE_MULTIPLIER:-3}" \
                FRCHECK_LAYER_STREAM_ENCODE="${FRCHECK_LAYER_STREAM_ENCODE:-1}" \
                FRCHECK_LAYER_ENCODE_COALESCE_US="${FRCHECK_LAYER_ENCODE_COALESCE_US:-0}" \
                FRCHECK_SEND_LANES_PER_PEER="${FRCHECK_SEND_LANES_PER_PEER:-12}" \
                FRCHECK_RECV_LANES_PER_PEER="${FRCHECK_RECV_LANES_PER_PEER:-12}" \
                FRCHECK_NET_PHYSICAL_CORES="${FRCHECK_NET_PHYSICAL_CORES:-0}" \
                FRCHECK_RDMA_CHUNK_MB="${FRCHECK_RDMA_CHUNK_MB:-64}" \
                FRCHECK_TRACE_INIT="${FRCHECK_TRACE_INIT:-0}" \
                FRCHECK_GDR="${FRCHECK_GDR:-1}" \
                FRCHECK_ASYNC_PARITY="${FRCHECK_ASYNC_PARITY:-1}" \
                FRCHECK_SAVE_PREFER_TORCH_PINNED="${FRCHECK_SAVE_PREFER_TORCH_PINNED:-0}" \
                "./$script" 0 save --train-iters 10 2>&1 | awk '{ print "[node0 rank=0]", $0; fflush(); }'
            exit "${PIPESTATUS[0]}"
        ) &
        local_pid=$!
        set +e
        wait "$local_pid"; local_rc=$?
        wait "$remote_pid"; remote_rc=$?
        set -e
        echo "[driver] local_exit=$local_rc remote_exit=$remote_rc"
        [[ "$local_rc" == 0 && "$remote_rc" == 0 ]]
    ) 2>&1 | tee "$scheme_log"
    return "${PIPESTATUS[0]}"
}

SIZE_PY=$(cat <<'SIZEPY'
import fnmatch
import os
import stat
import sys

node, scheme, root = sys.argv[1:4]
if not os.path.isdir(root):
    raise SystemExit(f"Checkpoint path does not exist: {root}")
iter_names = sorted(entry.name for entry in os.scandir(root) if entry.is_dir(follow_symlinks=False) and entry.name.startswith("iter_"))
if len(iter_names) != 1:
    raise SystemExit(f"Expected exactly one iter_* directory under {root}, found {len(iter_names)}: {iter_names}")
counts = {"all": [0, 0], "main": [0, 0], "selected": [0, 0]}
for directory, _, names in os.walk(root, followlinks=False):
    for name in names:
        filename = os.path.join(directory, name)
        try:
            info = os.stat(filename, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        size = info.st_size
        counts["all"][0] += size
        counts["all"][1] += 1
        is_main = "main" in name.lower()
        if is_main:
            counts["main"][0] += size
            counts["main"][1] += 1
        selected = fnmatch.fnmatchcase(name, "gemini_replicas_main_rank*.pt") if scheme in ("gemini2", "gemini3") else not is_main
        if selected:
            counts["selected"][0] += size
            counts["selected"][1] += 1
print(f"SIZE,node={node},scheme={scheme},iter={iter_names[0]},all_bytes={counts['all'][0]},all_files={counts['all'][1]},all_gib={counts['all'][0]/1073741824:.6f},main_bytes={counts['main'][0]},main_files={counts['main'][1]},main_gib={counts['main'][0]/1073741824:.6f},selected_bytes={counts['selected'][0]},selected_files={counts['selected'][1]},selected_gib={counts['selected'][0]/1073741824:.6f}")
SIZEPY
)

field_value() { local line=$1 field=$2; printf '%s\n' "$line" | awk -F"$field=" '{print $2}' | cut -d, -f1; }

collect_sizes() {
    local scheme=$1 checkpoint_path=${CHECKPOINT_PATHS[$1]}
    local node output python_command iter_name="" current_iter value
    local all_bytes=0 all_files=0 main_bytes=0 main_files=0 selected_bytes=0 selected_files=0
    for node in "${ALL_NODES[@]}"; do
        if [[ "$node" == node0 ]]; then
            output=$(python3 -c "$SIZE_PY" "$node" "$scheme" "$checkpoint_path") || return 1
        else
            printf -v python_command 'python3 -c %q %q %q %q' "$SIZE_PY" "$node" "$scheme" "$checkpoint_path"
            output=$(ssh_run "$node" "$python_command") || return 1
        fi
        summary "$output"
        current_iter=$(field_value "$output" iter)
        if [[ -z "$iter_name" ]]; then iter_name=$current_iter; elif [[ "$current_iter" != "$iter_name" ]]; then
            echo "Iteration directory mismatch: expected $iter_name, $node has $current_iter" >&2; return 1
        fi
        value=$(field_value "$output" all_bytes); all_bytes=$((all_bytes + value))
        value=$(field_value "$output" all_files); all_files=$((all_files + value))
        value=$(field_value "$output" main_bytes); main_bytes=$((main_bytes + value))
        value=$(field_value "$output" main_files); main_files=$((main_files + value))
        value=$(field_value "$output" selected_bytes); selected_bytes=$((selected_bytes + value))
        value=$(field_value "$output" selected_files); selected_files=$((selected_files + value))
    done
    summary "SIZE_TOTAL,scheme=$scheme,iter=$iter_name,all_bytes=$all_bytes,all_files=$all_files,all_gib=$(python3 -c "print(f'{$all_bytes / 1073741824:.6f}')"),main_bytes=$main_bytes,main_files=$main_files,main_gib=$(python3 -c "print(f'{$main_bytes / 1073741824:.6f}')"),selected_bytes=$selected_bytes,selected_files=$selected_files,selected_gib=$(python3 -c "print(f'{$selected_bytes / 1073741824:.6f}')")"
}

record_shm_available
summary "SWEEP,status=started,utc=$UTC_TIMESTAMP,schemes=$(IFS=:; echo "${schemes[*]}"),rdma_hca_profile=$RDMA_HCA_PROFILE"
overall_status=0
for scheme in "${schemes[@]}"; do
    summary "SCHEME,scheme=$scheme,status=starting,master_port=${MASTER_PORTS[$scheme]}"
    if ! cleanup_scheme "$scheme"; then
        summary "SCHEME,scheme=$scheme,status=failed,stage=pre_cleanup"; overall_status=1
        break
    fi
    if ! run_training "$scheme"; then
        summary "SCHEME,scheme=$scheme,status=failed,stage=training"; overall_status=1
        if ! cleanup_scheme "$scheme"; then
            summary "SCHEME,scheme=$scheme,status=failed,stage=failure_cleanup"
            break
        fi
        [[ "$CONTINUE_ON_ERROR" == 1 ]] && continue || break
    fi
    if ! collect_sizes "$scheme"; then
        summary "SCHEME,scheme=$scheme,status=failed,stage=size_validation"; overall_status=1
        if ! cleanup_scheme "$scheme"; then
            summary "SCHEME,scheme=$scheme,status=failed,stage=failure_cleanup"
            break
        fi
        [[ "$CONTINUE_ON_ERROR" == 1 ]] && continue || break
    fi
    if ! cleanup_scheme "$scheme"; then
        summary "SCHEME,scheme=$scheme,status=failed,stage=post_cleanup"; overall_status=1
        break
    fi
    summary "SCHEME,scheme=$scheme,status=success"
done
summary "SWEEP,status=$([[ $overall_status == 0 ]] && echo success || echo failed),utc=$(date -u +%Y%m%dT%H%M%SZ)"
exit "$overall_status"
