#!/bin/bash

set -uo pipefail

WORKDIR="/workspace/Megatron-LM"
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
SWEEP_TAG="${MODEL_SLUG}_inprocess_sweep"
HOSTS=(node1 node2 node3)
ALL_NODES=(node0 node1 node2 node3)
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"
INPROCESS_REPEAT="${INPROCESS_REPEAT:-10}"
DRY_RUN="${DRY_RUN:-0}"
RECOVERY_MODES_RAW="${RECOVERY_MODES-inprocess_sw inprocess inprocess2}"
TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-}"
GEMINI_ENV_VARS=(
    GEMINI_GDR
    GEMINI_GDR_MIRROR_MODE
    GEMINI_GDR_BATCH_WR
    GEMINI_MIRROR_CHUNK_MB
    GEMINI_REPLICAS_CHANNELS_PER_PEER
)

DEFAULT_SCHEMES=(gemini2 gemini3 frcheck eccheck ecnaive)
SELECTED_RECOVERY_MODES=()
declare -A MODE_LABELS=([inprocess_sw]=inprocess_sw [inprocess]=inprocess [inprocess2]=inprocess2)
declare -A MODE_OFFSETS=([save]=0 [inprocess_sw]=1 [inprocess]=2 [inprocess2]=3)
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
declare -A BASE_PORTS=(
    [gemini2]="${MASTER_PORT_GEMINI2:-6200}"
    [gemini3]="${MASTER_PORT_GEMINI3:-6240}"
    [frcheck]="${MASTER_PORT_FRCHECK:-6280}"
    [eccheck]="${MASTER_PORT_ECCHECK:-6320}"
    [ecnaive]="${MASTER_PORT_ECNAIVE:-6360}"
)

usage() {
    echo "Usage: $0 [--dry-run] [gemini2|gemini3|frcheck|eccheck|ecnaive ...]"
    echo "Environment: MODEL_SIZE=2.7B|7B|10B|14B|20B, LOG_DIR, RUN_TIMEOUT_SECONDS, DRY_RUN=0|1"
    echo "             RECOVERY_MODES, TRAIN_ENV_PREFIX, INPROCESS_REPEAT (positive integer, default: 10)"
    echo "             GEMINI_GDR, GEMINI_GDR_MIRROR_MODE, GEMINI_GDR_BATCH_WR,"
    echo "             GEMINI_MIRROR_CHUNK_MB, GEMINI_REPLICAS_CHANNELS_PER_PEER"
    echo "             SSH_USER, SSH_PORT, SSH_CONNECT_TIMEOUT, SSH_IDENTITY_FILE"
    echo "             MASTER_PORT_GEMINI2, MASTER_PORT_GEMINI3, MASTER_PORT_FRCHECK,"
    echo "             MASTER_PORT_ECCHECK, MASTER_PORT_ECNAIVE"
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

declare -A ALLOWED_RECOVERY_MODES=([inprocess_sw]=1 [inprocess]=1 [inprocess2]=1)
declare -A SEEN_RECOVERY_MODES=()
read -r -a requested_recovery_modes <<< "${RECOVERY_MODES_RAW//,/ }"
for mode in "${requested_recovery_modes[@]}"; do
    [[ -n "${ALLOWED_RECOVERY_MODES[$mode]+x}" ]] || { echo "Invalid recovery mode: $mode" >&2; exit 2; }
    if [[ -z "${SEEN_RECOVERY_MODES[$mode]+x}" ]]; then
        SELECTED_RECOVERY_MODES+=("$mode")
        SEEN_RECOVERY_MODES[$mode]=1
    fi
done
[[ ${#SELECTED_RECOVERY_MODES[@]} -gt 0 ]] || { echo "RECOVERY_MODES must select at least one recovery mode." >&2; exit 2; }
build_command_env_prefix() {
    local scheme=$1 name value prefix="env ${TRAIN_ENV_PREFIX:+$TRAIN_ENV_PREFIX }"
    if [[ "$scheme" == gemini2 || "$scheme" == gemini3 ]]; then
        printf -v value '%q' "${GEMINI_GDR:-0}"
        prefix+="GEMINI_GDR=$value "
        for name in "${GEMINI_ENV_VARS[@]:1}"; do
            if [[ -v $name ]]; then
                printf -v value '%q' "${!name}"
                prefix+="$name=$value "
            fi
        done
    fi
    printf '%s' "$prefix"
}

TRAIN_ENV_COMMAND_PREFIX="env ${TRAIN_ENV_PREFIX:+$TRAIN_ENV_PREFIX }"
FRCHECK_HW2_ASYNC_OFF=0
if [[ "${FRCHECK_RECOVERY_ASYNC_PARITY:-1}" == 0 || " $TRAIN_ENV_PREFIX " == *" FRCHECK_RECOVERY_ASYNC_PARITY=0 "* ]]; then
    FRCHECK_HW2_ASYNC_OFF=1
fi

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1." >&2; exit 2; }
[[ "$RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "RUN_TIMEOUT_SECONDS must be a positive integer." >&2; exit 2; }
[[ "$INPROCESS_REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "INPROCESS_REPEAT must be a positive integer." >&2; exit 2; }
for scheme in "${schemes[@]}"; do
    [[ "${BASE_PORTS[$scheme]}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid base port for $scheme." >&2; exit 2; }
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

training_processes() {
    local node=$1 command="pgrep -af '[t]orchrun|pretrain_gpt[.]py'"
    if [[ "$node" == node0 ]]; then
        bash -lc "$command" || true
    else
        ssh_run "$node" "$command" || true
    fi
}

GPU_CONTEXT_PY=$(cat <<'PYCODE'
import os
import subprocess
import sys

result = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print((result.stderr or result.stdout).strip().replace('\n', ';')[:500])
    sys.exit(2)
pids = sorted({line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()}, key=int)
invisible = [pid for pid in pids if not os.path.exists(f'/proc/{pid}')]
print(':'.join(invisible))
sys.exit(1 if invisible else 0)
PYCODE
)

gpu_context_check() {
    local node=$1 command
    printf -v command 'python3 -c %q' "$GPU_CONTEXT_PY"
    if [[ "$node" == node0 ]]; then bash -lc "$command"; else ssh_run "$node" "$command"; fi
}

preflight() {
    local scheme host script checkpoint_path output
    [[ -d "$WORKDIR" ]] || { echo "Work directory not found: $WORKDIR" >&2; return 1; }
    command -v timeout >/dev/null || { echo "timeout command not found." >&2; return 1; }
    command -v setsid >/dev/null || { echo "setsid command not found." >&2; return 1; }
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
    for host in "${HOSTS[@]}"; do
        ssh_run "$host" "true" >/dev/null || { echo "SSH check failed for $host" >&2; return 1; }
        for scheme in "${schemes[@]}"; do
            script=${SCRIPTS[$scheme]}
            ssh_run "$host" "test -d '$WORKDIR' && test -f '$WORKDIR/$script'" || {
                echo "Remote path check failed on $host for $WORKDIR/$script" >&2
                return 1
            }
        done
    done
    for host in "${ALL_NODES[@]}"; do
        output=$(training_processes "$host")
        if [[ -n "$output" ]]; then
            echo "Existing training process found on $host:" >&2
            echo "$output" >&2
            return 1
        fi
        output=$(gpu_context_check "$host" 2>&1); rc=$?
        if [[ $rc -eq 1 ]]; then
            echo "GPU_CONTEXT,node=$host,status=unowned_or_invisible,pids=${output:-unknown}" >&2
            return 1
        elif [[ $rc -ne 0 ]]; then
            echo "GPU_CONTEXT,node=$host,status=check_failed,exit=$rc,detail=${output:-unknown}" >&2
            return 1
        fi
    done
}

preflight || exit 1

print_dry_run() {
    local scheme mode script path base port command_env_prefix
    echo "Dry-run command plan: recovery_modes=$(IFS=:; echo "${SELECTED_RECOVERY_MODES[*]}") train_env_prefix=${TRAIN_ENV_PREFIX:-<empty>}"
    for scheme in "${schemes[@]}"; do
        script=${SCRIPTS[$scheme]}; path=${CHECKPOINT_PATHS[$scheme]}; base=${BASE_PORTS[$scheme]}
        command_env_prefix=$(build_command_env_prefix "$scheme")
        echo "SCHEME=$scheme PATH=$path BASE_PORT=$base"
        if [[ "$scheme" == gemini2 || "$scheme" == gemini3 ]]; then
            echo "  gemini_config: ${command_env_prefix#env ${TRAIN_ENV_PREFIX:+$TRAIN_ENV_PREFIX }}"
        fi
        for mode in save "${SELECTED_RECOVERY_MODES[@]}"; do
            port=$((base + MODE_OFFSETS[$mode]))
            if [[ "$scheme" == gemini2 && "$mode" == inprocess2 ]]; then
                echo "  node0-only expected unsupported: ${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port ./$script 0 $mode --train-iters 2"
            elif [[ "$scheme" == frcheck && "$mode" == inprocess2 && "$FRCHECK_HW2_ASYNC_OFF" == 1 ]]; then
                echo "  not required: FRCheck HW2 with recovery async parity off"
            else
                echo "  four-node: ${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port ./$script {R} $mode --train-iters 2"
            fi
        done
        echo "  cleanup before save and after all recovery modes: $path on node0,node1,node2,node3"
    done
}

if [[ "$DRY_RUN" == 1 ]]; then
    print_dry_run
    exit 0
fi

UTC_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="${LOG_DIR:-$WORKDIR/logs/${SWEEP_TAG}/$UTC_TIMESTAMP}"
python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)' "$LOG_DIR" || exit 1
[[ -d "$LOG_DIR" ]] || { echo "Failed to create log directory: $LOG_DIR" >&2; exit 1; }
SUMMARY_LOG="$LOG_DIR/summary.log"
: >"$SUMMARY_LOG"
summary() { printf '%s\n' "$*" | tee -a "$SUMMARY_LOG"; }

utc_now() { date -u +%Y%m%dT%H%M%SZ; }
epoch_now() { date -u +%s; }

record_shm_available() {
    local node output command
    command="python3 -c 'import os; s=os.statvfs(\"/dev/shm\"); print(s.f_bavail*s.f_frsize)'"
    for node in "${ALL_NODES[@]}"; do
        if [[ "$node" == node0 ]]; then output=$(bash -lc "$command" 2>&1); rc=$?; else output=$(ssh_run "$node" "$command" 2>&1); rc=$?; fi
        if [[ $rc -eq 0 ]]; then
            summary "SHM_AVAILABLE,node=$node,status=success,bytes=$output"
        else
            summary "SHM_AVAILABLE,node=$node,status=failed,exit=$rc,detail=${output//$'\n'/ }"
        fi
    done
}

cleanup_scheme() {
    local scheme=$1 phase=$2 checkpoint_path=${CHECKPOINT_PATHS[$1]} node failed=0
    if ! validate_checkpoint_path "$checkpoint_path"; then
        summary "CLEANUP,scheme=$scheme,phase=$phase,node=all,status=failed,path=$checkpoint_path,reason=unsafe_path"
        return 1
    fi
    if ! rm -rf -- "$checkpoint_path"; then failed=1; fi
    for node in "${HOSTS[@]}"; do
        ssh_run "$node" "rm -rf -- '$checkpoint_path'" || failed=1
    done
    for node in "${ALL_NODES[@]}"; do
        if [[ "$node" == node0 ]]; then test ! -e "$checkpoint_path"; rc=$?; else ssh_run "$node" "test ! -e '$checkpoint_path'"; rc=$?; fi
        if [[ $rc -eq 0 ]]; then
            summary "CLEANUP,scheme=$scheme,phase=$phase,node=$node,status=success,path=$checkpoint_path"
        else
            summary "CLEANUP,scheme=$scheme,phase=$phase,node=$node,status=failed,path=$checkpoint_path"
            failed=1
        fi
    done
    [[ $failed -eq 0 ]]
}

OWNED_PROCESS_PY=$(cat <<'PYCODE'
import os
import signal
import sys

tag, action = sys.argv[1:3]
found = []
for name in os.listdir('/proc'):
    if not name.isdigit() or int(name) == os.getpid():
        continue
    try:
        env = open(f'/proc/{name}/environ', 'rb').read().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    values = dict(item.split(b'=', 1) for item in env if b'=' in item)
    if values.get(b'SWEEP_RUN_TAG', b'').decode(errors='replace') != tag:
        continue
    found.append(int(name))
if action != 'list':
    sig = signal.SIGTERM if action == 'term' else signal.SIGKILL
    groups = set()
    for pid in found:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    for group in groups:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            pass
print(' '.join(map(str, found)))
PYCODE
)

owned_processes() {
    local node=$1 tag=$2 action=${3:-list} command
    printf -v command 'python3 -c %q %q %q' "$OWNED_PROCESS_PY" "$tag" "$action"
    if [[ "$node" == node0 ]]; then bash -lc "$command"; else ssh_run "$node" "$command"; fi
}

terminate_owned_run() {
    local tag=$1 node
    summary "TERMINATE,tag=$tag,status=starting"
    for node in "${ALL_NODES[@]}"; do owned_processes "$node" "$tag" term >/dev/null 2>&1 || true; done
    sleep 3
    for node in "${ALL_NODES[@]}"; do owned_processes "$node" "$tag" kill >/dev/null 2>&1 || true; done
    sleep 1
    for node in "${ALL_NODES[@]}"; do
        remaining=$(owned_processes "$node" "$tag" list 2>/dev/null || true)
        summary "TERMINATE,tag=$tag,node=$node,status=$([[ -z "$remaining" ]] && echo success || echo failed),remaining=${remaining:-none}"
    done
}

check_before_run() {
    local scheme=$1 mode=$2 previous_tag=${3:-} node output owned rc detail truncated failed=0
    if [[ -n "$previous_tag" ]]; then
        for node in "${ALL_NODES[@]}"; do
            owned=$(owned_processes "$node" "$previous_tag" list 2>/dev/null || true)
            if [[ -n "$owned" ]]; then
                summary "RESIDUAL,scheme=$scheme,mode=$mode,node=$node,status=owned_found,tag=$previous_tag,pids=${owned// /:}"
                terminate_owned_run "$previous_tag"
                break
            fi
        done
    fi
    for node in "${ALL_NODES[@]}"; do
        output=$(training_processes "$node")
        if [[ -n "$output" ]]; then
            detail=${output//$'\n'/;}
            truncated=false
            if (( ${#detail} > 500 )); then detail=${detail:0:500}; truncated=true; fi
            summary "RESIDUAL,scheme=$scheme,mode=$mode,node=$node,status=found,detail=$detail,truncated=$truncated"
            failed=1
        else
            summary "RESIDUAL,scheme=$scheme,mode=$mode,node=$node,status=clear"
        fi
        output=$(gpu_context_check "$node" 2>&1); rc=$?
        if [[ $rc -eq 1 ]]; then
            summary "GPU_CONTEXT,scheme=$scheme,mode=$mode,node=$node,status=unowned_or_invisible,pids=${output:-unknown}"
            failed=1
        elif [[ $rc -ne 0 ]]; then
            detail=${output//$'\n'/;}
            truncated=false
            if (( ${#detail} > 500 )); then detail=${detail:0:500}; truncated=true; fi
            summary "GPU_CONTEXT,scheme=$scheme,mode=$mode,node=$node,status=check_failed,exit=$rc,detail=$detail,truncated=$truncated"
            failed=1
        else
            summary "GPU_CONTEXT,scheme=$scheme,mode=$mode,node=$node,status=clear"
        fi
    done
    [[ $failed -eq 0 ]]
}

run_four_nodes() {
    local scheme=${1:?} mode=${2:?} port=${3:?} log=${4:?} tag=${5:?}
    local self_test=${RUN_FOUR_NODES_SELF_TEST:-0}

    if [[ "$self_test" == 1 && -z "${RUN_FOUR_NODES_SELF_TEST_CASE:-}" ]]; then
        local success_output failure_output success_rc failure_rc failure_start failure_elapsed
        local success_iteration
        for success_iteration in {1..20}; do
            if success_output=$(RUN_FOUR_NODES_SELF_TEST_CASE=success run_four_nodes self_test self_test 1 /dev/null "self_test_success_$success_iteration" 2>&1); then
                success_rc=0
            else
                success_rc=$?
            fi
            printf '%s\n' "$success_output"
            [[ $success_rc -eq 0 ]] || {
                echo "SELF_TEST,status=failed,case=success,iteration=$success_iteration,exit=$success_rc" >&2
                return 1
            }
            [[ "$success_output" != *"task_exit=127"* ]] || {
                echo "SELF_TEST,status=failed,case=success_127,iteration=$success_iteration" >&2
                return 1
            }
            [[ $(grep -c 'task_exit=0' <<<"$success_output") -eq 4 ]] || {
                echo "SELF_TEST,status=failed,case=success_count,iteration=$success_iteration" >&2
                return 1
            }
        done
        failure_start=$(date +%s)
        if failure_output=$(RUN_FOUR_NODES_SELF_TEST_CASE=failure run_four_nodes self_test self_test 1 /dev/null self_test_failure 2>&1); then
            failure_rc=0
        else
            failure_rc=$?
        fi
        failure_elapsed=$(( $(date +%s) - failure_start ))
        printf '%s\n' "$failure_output"
        [[ $failure_rc -eq 23 ]] || { echo "SELF_TEST,status=failed,case=failure,exit=$failure_rc" >&2; return 1; }
        (( failure_elapsed <= 8 )) || { echo "SELF_TEST,status=failed,case=fail_fast,duration_seconds=$failure_elapsed" >&2; return 1; }
        [[ "$success_output" == *"[node0 rank=0]"* && "$success_output" == *"[node3 rank=3]"* ]] || {
            echo "SELF_TEST,status=failed,case=prefix" >&2
            return 1
        }
        echo "SELF_TEST,status=success,success_exit=$success_rc,failure_exit=$failure_rc,failure_duration_seconds=$failure_elapsed"
        return 0
    fi

    local script workdir remote_command local_command timeout_seconds command_env_prefix
    local ssh_user ssh_port ssh_timeout ssh_key self_test_case
    workdir=${WORKDIR:-$PWD}
    timeout_seconds=${RUN_TIMEOUT_SECONDS:-30}
    ssh_user=${SSH_USER:-root}
    ssh_port=${SSH_PORT:-22}
    ssh_timeout=${SSH_CONNECT_TIMEOUT:-10}
    ssh_key=${SSH_IDENTITY_FILE:-}
    self_test_case=${RUN_FOUR_NODES_SELF_TEST_CASE:-}

    if [[ "$self_test" == 1 ]]; then
        remote_command=$(cat <<'EOF'
rank={R}; echo "self-test rank=$rank"; if [[ "$RUN_FOUR_NODES_SELF_TEST_CASE" == failure && "$rank" == 2 ]]; then exit 23; fi; if [[ "$RUN_FOUR_NODES_SELF_TEST_CASE" == failure ]]; then sleep 20; else sleep 0.05; fi
EOF
)
        local_command=${remote_command//\{R\}/0}
        script=self_test
    else
        script=${SCRIPTS[$scheme]:?}
        command_env_prefix=$(build_command_env_prefix "$scheme")
        remote_command="${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port SWEEP_RUN_TAG=$tag SWEEP_SCRIPT=$script SWEEP_MODE=$mode ./$script {R} $mode --train-iters 2"
        local_command="${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port SWEEP_RUN_TAG=$tag SWEEP_SCRIPT=$script SWEEP_MODE=$mode ./$script 0 $mode --train-iters 2"
    fi

    {
        echo "COMMAND,scope=remote,nodes=node1:node2:node3,scheme=$scheme,mode=$mode,repeat=$INPROCESS_REPEAT,train_iters=2,port=$port,command=$remote_command"
        echo "COMMAND,scope=local,node=node0,scheme=$scheme,mode=$mode,repeat=$INPROCESS_REPEAT,train_iters=2,port=$port,command=$local_command"
    } >"$log"

    run_four_nodes_launcher() {
        local host=${1:?} rank=${2:?} launcher_workdir=${3:?} command_template=${4:?}
        local launcher_ssh_user=${5:?} launcher_ssh_port=${6:?} launcher_ssh_timeout=${7:?}
        local launcher_ssh_key=$8 launcher_self_test=$9 host_command rc quoted_command
        local -a ssh_options
        host_command=${command_template//\{R\}/$rank}
        set -o pipefail
        if [[ "$launcher_self_test" == 1 ]]; then
            RUN_FOUR_NODES_SELF_TEST_CASE=${RUN_FOUR_NODES_SELF_TEST_CASE:?} bash -lc "$host_command" 2>&1 |
                awk -v prefix="[$host rank=$rank]" '{ print prefix, $0; fflush(); }'
        else
            ssh_options=(-n -p "$launcher_ssh_port" -o BatchMode=yes -o ConnectTimeout="$launcher_ssh_timeout" -o StrictHostKeyChecking=accept-new)
            [[ -z "$launcher_ssh_key" ]] || ssh_options+=(-i "$launcher_ssh_key")
            printf -v quoted_command 'export PYTHONUNBUFFERED=1; cd %q && %s' "$launcher_workdir" "$host_command"
            printf -v quoted_command %q "$quoted_command"
            ssh "${ssh_options[@]}" "${launcher_ssh_user}@${host}" "bash -lc $quoted_command" 2>&1 |
                awk -v prefix="[$host rank=$rank]" '{ print prefix, $0; fflush(); }'
        fi
        rc=${PIPESTATUS[0]}
        echo "[$host rank=$rank] command_exit=$rc"
        return "$rc"
    }

    run_four_nodes_local_launcher() {
        local launcher_workdir=${1:?} command=${2:?} launcher_self_test=$3 rc
        set -o pipefail
        cd "$launcher_workdir" || return 1
        if [[ "$launcher_self_test" == 1 ]]; then
            RUN_FOUR_NODES_SELF_TEST_CASE=${RUN_FOUR_NODES_SELF_TEST_CASE:?} bash -lc "$command" 2>&1 |
                awk '{ print "[node0 rank=0]", $0; fflush(); }'
        else
            bash -lc "$command" 2>&1 | awk '{ print "[node0 rank=0]", $0; fflush(); }'
        fi
        rc=${PIPESTATUS[0]}
        echo "[node0 rank=0] command_exit=$rc"
        return "$rc"
    }

    run_four_nodes_driver() {
        local driver_workdir=${1:?} driver_remote_command=${2:?} driver_local_command=${3:?}
        local driver_ssh_user=${4:?} driver_ssh_port=${5:?} driver_ssh_timeout=${6:?}
        local driver_ssh_key=$7 driver_self_test=$8 rank host rc remaining launcher_pid
        local active_pid collected_any
        local -a pids=()
        local -A pending_ranks=() active_pids=()

        stop_launchers() {
            local signal=$1 launcher_pid launcher_pgid
            for launcher_pid in "${pids[@]}"; do
                launcher_pgid=$(ps -o pgid= -p "$launcher_pid" 2>/dev/null) || continue
                launcher_pgid=${launcher_pgid//[[:space:]]/}
                [[ "$launcher_pgid" == "$launcher_pid" ]] || continue
                kill -"$signal" -- "-$launcher_pgid" 2>/dev/null || true
            done
        }
        trap 'stop_launchers TERM; sleep 1; stop_launchers KILL; exit 143' TERM INT

        for rank in 1 2 3; do
            host="node$rank"
            setsid bash -c 'run_four_nodes_launcher "$@"' _ \
                "$host" "$rank" "$driver_workdir" "$driver_remote_command" \
                "$driver_ssh_user" "$driver_ssh_port" "$driver_ssh_timeout" "$driver_ssh_key" "$driver_self_test" &
            pids+=("$!")
            pending_ranks[${pids[-1]}]=$rank
        done
        setsid bash -c 'run_four_nodes_local_launcher "$@"' _ \
            "$driver_workdir" "$driver_local_command" "$driver_self_test" &
        pids+=("$!")
        pending_ranks[${pids[-1]}]=0

        remaining=${#pids[@]}
        while (( remaining > 0 )); do
            active_pids=()
            while IFS= read -r active_pid; do
                [[ -n "$active_pid" ]] && active_pids[$active_pid]=1
            done < <(jobs -pr)
            collected_any=0
            for launcher_pid in "${!pending_ranks[@]}"; do
                [[ -v active_pids[$launcher_pid] ]] && continue
                if wait "$launcher_pid"; then rc=0; else rc=$?; fi
                rank=${pending_ranks[$launcher_pid]}
                unset 'pending_ranks[$launcher_pid]'
                ((remaining--))
                collected_any=1
                echo "[driver] task_exit=$rc rank=$rank pid=$launcher_pid remaining=$remaining"
                if [[ $rc -ne 0 ]]; then
                    stop_launchers TERM
                    sleep 1
                    stop_launchers KILL
                    for launcher_pid in "${!pending_ranks[@]}"; do
                        wait "$launcher_pid" 2>/dev/null || true
                        unset 'pending_ranks[$launcher_pid]'
                    done
                    return "$rc"
                fi
            done
            (( collected_any == 1 || remaining == 0 )) || sleep 0.05
        done
        return 0
    }

    export -f run_four_nodes_launcher run_four_nodes_local_launcher run_four_nodes_driver
    export RUN_FOUR_NODES_SELF_TEST_CASE="$self_test_case"
    timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
        bash -c 'run_four_nodes_driver "$@"' _ \
        "$workdir" "$remote_command" "$local_command" \
        "$ssh_user" "$ssh_port" "$ssh_timeout" "$ssh_key" "$self_test" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 && "$self_test" != 1 ]]; then terminate_owned_run "$tag"; fi
    return "$rc"
}

run_expected_unsupported() {
    local scheme=$1 mode=$2 port=$3 log=$4 tag=$5 script=${SCRIPTS[$1]} command command_env_prefix
    command_env_prefix=$(build_command_env_prefix "$scheme")
    command="${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port SWEEP_RUN_TAG=$tag SWEEP_SCRIPT=$script SWEEP_MODE=$mode ./$script 0 $mode --train-iters 2"
    echo "COMMAND,scope=local,node=node0,scheme=$scheme,mode=$mode,repeat=$INPROCESS_REPEAT,train_iters=2,port=$port,command=$command" >"$log"
    timeout --signal=TERM --kill-after=10s "$RUN_TIMEOUT_SECONDS" bash -c 'cd "$1" && bash -lc "$2"' _ "$WORKDIR" "$command" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 124 || $rc -eq 137 || $rc -eq 143 ]]; then terminate_owned_run "$tag"; fi
    return "$rc"
}

validate_checkpoint() {
    local scheme=$1 path=${CHECKPOINT_PATHS[$1]} node command output expected="iter_0000001" failed=0
    command="python3 -c 'import os,sys; p=sys.argv[1]; print(\"|\".join(sorted(e.name for e in os.scandir(p) if e.is_dir(follow_symlinks=False) and e.name.startswith(\"iter_\"))))' '$path'"
    for node in "${ALL_NODES[@]}"; do
        if [[ "$node" == node0 ]]; then output=$(bash -lc "$command" 2>&1); rc=$?; else output=$(ssh_run "$node" "$command" 2>&1); rc=$?; fi
        if [[ $rc -ne 0 || "$output" != "$expected" ]]; then
            summary "ITER_VALIDATION,scheme=$scheme,node=$node,status=failed,exit=$rc,expected=$expected,found=${output//$'\n'/ }"
            failed=1
        else
            summary "ITER_VALIDATION,scheme=$scheme,node=$node,status=success,iter=$output"
        fi
    done
    [[ $failed -eq 0 ]]
}

count_log_matches() {
    local pattern=$1 log=$2
    grep -c -- "$pattern" "$log" 2>/dev/null || true
}

record_shm_available
summary "SWEEP,status=started,utc=$UTC_TIMESTAMP,schemes=$(IFS=:; echo "${schemes[*]}"),recovery_modes=$(IFS=:; echo "${SELECTED_RECOVERY_MODES[*]}"),train_env_prefix=${TRAIN_ENV_PREFIX:-<empty>},repeat=$INPROCESS_REPEAT,train_iters=2,timeout_seconds=$RUN_TIMEOUT_SECONDS,log_dir=$LOG_DIR"
overall_status=0

for scheme in "${schemes[@]}"; do
    script=${SCRIPTS[$scheme]}
    base=${BASE_PORTS[$scheme]}
    command_env_prefix=$(build_command_env_prefix "$scheme")
    scheme_failed=0
    summary "SCHEME,scheme=$scheme,status=started,utc=$(utc_now),base_port=$base,path=${CHECKPOINT_PATHS[$scheme]}"

    pre_cleanup_ok=1
    if ! cleanup_scheme "$scheme" pre_save; then overall_status=1; scheme_failed=1; pre_cleanup_ok=0; fi

    save_log="$LOG_DIR/${scheme}_save.log"
    save_port=$((base + MODE_OFFSETS[save]))
    save_tag="${SWEEP_TAG}_${UTC_TIMESTAMP}_${scheme}_save_${save_port}"
    save_start_utc=$(utc_now); save_start_epoch=$(epoch_now)
    save_rc=1; iter_status=failed
    save_planned="${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$save_port SWEEP_RUN_TAG=$save_tag SWEEP_SCRIPT=$script SWEEP_MODE=save ./$script {R} save --train-iters 2"
    if [[ $pre_cleanup_ok -eq 0 ]]; then
        printf '%s\n%s\n' "COMMAND,scope=four_nodes,scheme=$scheme,mode=save,repeat=$INPROCESS_REPEAT,train_iters=2,port=$save_port,command=$save_planned" "COMMAND_SKIPPED,reason=pre_cleanup_failed" >"$save_log"
    elif check_before_run "$scheme" save; then
        run_four_nodes "$scheme" save "$save_port" "$save_log" "$save_tag"; save_rc=$?
        if [[ $save_rc -eq 0 ]] && validate_checkpoint "$scheme"; then iter_status=success; fi
    else
        printf '%s\n%s\n' "COMMAND,scope=four_nodes,scheme=$scheme,mode=save,repeat=$INPROCESS_REPEAT,train_iters=2,port=$save_port,command=$save_planned" "COMMAND_SKIPPED,reason=training_process_found" >"$save_log"
    fi
    save_end_epoch=$(epoch_now); save_end_utc=$(utc_now)
    if [[ $save_rc -eq 0 && "$iter_status" == success ]]; then
        save_status=success
    else
        save_status=failed; overall_status=1; scheme_failed=1
    fi
    summary "SAVE,scheme=$scheme,status=$save_status,exit=$save_rc,iter_status=$iter_status,port=$save_port,start_utc=$save_start_utc,end_utc=$save_end_utc,duration_seconds=$((save_end_epoch-save_start_epoch)),log=$save_log"

    previous_tag=$save_tag
    if [[ "$save_status" == success ]]; then
        for mode in "${SELECTED_RECOVERY_MODES[@]}"; do
            port=$((base + MODE_OFFSETS[$mode]))
            log="$LOG_DIR/${scheme}_${MODE_LABELS[$mode]}.log"
            tag="${SWEEP_TAG}_${UTC_TIMESTAMP}_${scheme}_${mode}_${port}"
            start_utc=$(utc_now); start_epoch=$(epoch_now)
            rc=1; status=failed; target_run_count=0; run10_count=0; recovery_forward_count=0

            if [[ "$scheme" == frcheck && "$mode" == inprocess2 && "$FRCHECK_HW2_ASYNC_OFF" == 1 ]]; then
                printf '%s\n' "COMMAND_SKIPPED,reason=frcheck_hw2_async_off_not_required" >"$log"
                rc=0
                status=not_required
            elif ! check_before_run "$scheme" "$mode" "$previous_tag"; then
                planned="${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port SWEEP_RUN_TAG=$tag SWEEP_SCRIPT=$script SWEEP_MODE=$mode ./$script {R} $mode --train-iters 2"
                printf '%s\n%s\n' "COMMAND,scope=four_nodes,scheme=$scheme,mode=$mode,repeat=$INPROCESS_REPEAT,train_iters=2,port=$port,command=$planned" "COMMAND_SKIPPED,reason=training_process_found" >"$log"
                overall_status=1; scheme_failed=1
            elif [[ "$scheme" == gemini2 && "$mode" == inprocess2 ]]; then
                run_expected_unsupported "$scheme" "$mode" "$port" "$log" "$tag"; rc=$?
                if [[ $rc -eq 2 ]] && grep -Eiq 'cannot recover|unsupported' "$log"; then
                    status=expected_unsupported
                else
                    overall_status=1; scheme_failed=1
                fi
            else
                run_four_nodes "$scheme" "$mode" "$port" "$log" "$tag"; rc=$?
                target_run_count=$(count_log_matches "FT in-process recovery benchmark: run=$INPROCESS_REPEAT/$INPROCESS_REPEAT" "$log")
                run10_count=$target_run_count
                recovery_forward_count=$(count_log_matches ' forward max: forward_failed_max_s=' "$log")
                if [[ $recovery_forward_count -lt $INPROCESS_REPEAT ]]; then
                    legacy_recovery_forward_count=$(count_log_matches 'recovery-to-forward' "$log")
                    if [[ $legacy_recovery_forward_count -gt $recovery_forward_count ]]; then
                        recovery_forward_count=$legacy_recovery_forward_count
                    fi
                fi
                if [[ $rc -eq 0 && $target_run_count -ge 1 && $recovery_forward_count -ge $INPROCESS_REPEAT ]]; then
                    status=success
                else
                    overall_status=1; scheme_failed=1
                fi
            fi
            previous_tag=$tag
            end_epoch=$(epoch_now); end_utc=$(utc_now)
            summary "RUN,scheme=$scheme,mode=$mode,status=$status,exit=$rc,target_run_count=$target_run_count,run10_count=$run10_count,recovery_forward_count=$recovery_forward_count,port=$port,start_utc=$start_utc,end_utc=$end_utc,duration_seconds=$((end_epoch-start_epoch)),log=$log"
        done
    else
        for mode in "${SELECTED_RECOVERY_MODES[@]}"; do
            port=$((base + MODE_OFFSETS[$mode]))
            log="$LOG_DIR/${scheme}_${MODE_LABELS[$mode]}.log"
            planned="${command_env_prefix}PRINT_CMD=0 FT_INPROCESS_RECOVERY_REPEAT=$INPROCESS_REPEAT MASTER_PORT=$port SWEEP_RUN_TAG=not_run SWEEP_SCRIPT=$script SWEEP_MODE=$mode ./$script {R} $mode --train-iters 2"
            printf '%s\n%s\n' "COMMAND,scope=four_nodes,scheme=$scheme,mode=$mode,repeat=$INPROCESS_REPEAT,train_iters=2,port=$port,command=$planned" "COMMAND_SKIPPED,reason=save_failed" >"$log"
            summary "RUN,scheme=$scheme,mode=$mode,status=failed,exit=not_run,target_run_count=0,run10_count=0,recovery_forward_count=0,port=$port,start_utc=$(utc_now),end_utc=$(utc_now),duration_seconds=0,log=$log,reason=save_failed"
        done
    fi

    if ! cleanup_scheme "$scheme" post_runs; then overall_status=1; scheme_failed=1; fi
    summary "SCHEME,scheme=$scheme,status=$([[ $scheme_failed -eq 0 ]] && echo success || echo failed),utc=$(utc_now)"
done

summary "SWEEP,status=$([[ $overall_status -eq 0 ]] && echo success || echo failed),utc=$(utc_now),exit=$overall_status,log_dir=$LOG_DIR"
exit "$overall_status"
