#!/bin/bash

set -uo pipefail

# Keep hosts and their torchrun node ranks aligned by array index.
HOSTS=(node1 node2 node3)
NODE_RANKS=(1 2 3)

SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/workspace/Megatron-LM}"

usage() {
    echo "Usage: $0 [-r] [-q] [-w remote_workdir] -c \"command\""
    echo
    echo "Examples:"
    echo "  $0 -c \"chmod 777 -R /workspace/Megatron-LM\""
    echo "  $0 -q -c \"chmod 777 -R /workspace/Megatron-LM\""
    echo "  $0 -r -w /workspace/Megatron-LM -c \"./pre-tests/gpt2/para_a800/20B-4nodes/test_train.sh {R} 0 1 2 3 4 5 6 7 save\""
    echo
    echo "Environment overrides:"
    echo "  SSH_USER, SSH_PORT, SSH_CONNECT_TIMEOUT, SSH_IDENTITY_FILE, REMOTE_WORKDIR"
}

COMMAND=""
NODE_RANK_MODE=0
OUTPUT_ENABLED=1
while getopts ":c:w:rqh" option; do
    case "$option" in
        c)
            COMMAND="$OPTARG"
            ;;
        w)
            REMOTE_WORKDIR="$OPTARG"
            ;;
        r)
            NODE_RANK_MODE=1
            ;;
        q)
            OUTPUT_ENABLED=0
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage >&2
            exit 2
            ;;
        \?)
            echo "Unknown option: -$OPTARG" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$REMOTE_WORKDIR" ]]; then
    echo "Remote working directory must not be empty." >&2
    exit 2
fi

if [[ "$REMOTE_WORKDIR" != /* ]]; then
    echo "Remote working directory must be an absolute path: $REMOTE_WORKDIR" >&2
    exit 2
fi

if [[ -z "$COMMAND" ]]; then
    echo "A remote command must be provided with -c." >&2
    usage >&2
    exit 2
fi

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "HOSTS must contain at least one target." >&2
    exit 2
fi

if [[ "$NODE_RANK_MODE" -eq 1 ]]; then
    if [[ ${#HOSTS[@]} -ne ${#NODE_RANKS[@]} ]]; then
        echo "HOSTS and NODE_RANKS must have the same number of entries." >&2
        exit 2
    fi

    if [[ "$COMMAND" != *'{R}'* ]]; then
        echo 'Node-rank mode requires the {R} placeholder.' >&2
        echo 'Example: -r -c "./test_train.sh {R} 0 1 2 3 4 5 6 7 save"' >&2
        exit 2
    fi

    declare -A SEEN_NODE_RANKS=()
    for node_rank in "${NODE_RANKS[@]}"; do
        if [[ ! "$node_rank" =~ ^[0-9]+$ ]]; then
            echo "Invalid node rank: $node_rank" >&2
            exit 2
        fi
        if [[ -n "${SEEN_NODE_RANKS[$node_rank]+x}" ]]; then
            echo "Duplicate node rank: $node_rank" >&2
            exit 2
        fi
        SEEN_NODE_RANKS[$node_rank]=1
    done
fi

SSH_OPTIONS=(
    -n
    -p "$SSH_PORT"
    -o BatchMode=yes
    -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    -o StrictHostKeyChecking=accept-new
)

if [[ -n "$SSH_IDENTITY_FILE" ]]; then
    if [[ ! -f "$SSH_IDENTITY_FILE" ]]; then
        echo "SSH identity file not found: $SSH_IDENTITY_FILE" >&2
        exit 2
    fi
    SSH_OPTIONS+=(-i "$SSH_IDENTITY_FILE")
fi

pids=()

terminate_children() {
    if [[ ${#pids[@]} -gt 0 ]]; then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

for index in "${!HOSTS[@]}"; do
    host="${HOSTS[$index]}"
    node_rank=""
    host_command="$COMMAND"
    prefix="[$host]"
    if [[ "$NODE_RANK_MODE" -eq 1 ]]; then
        node_rank="${NODE_RANKS[$index]}"
        host_command="${COMMAND//\{R\}/$node_rank}"
        prefix="[$host rank=$node_rank]"
    fi
    printf -v workdir_command 'export PYTHONUNBUFFERED=1; cd %q && %s' "$REMOTE_WORKDIR" "$host_command"
    remote_command=$(printf '%q' "$workdir_command")
    if [[ "$OUTPUT_ENABLED" -eq 1 ]]; then
        echo "$prefix starting: $host_command"
    fi
    (
        set -o pipefail
        if [[ "$OUTPUT_ENABLED" -eq 1 ]]; then
            ssh "${SSH_OPTIONS[@]}" "${SSH_USER}@${host}" \
                "bash -lc $remote_command" 2>&1 |
                awk -v prefix="$prefix" '{ print prefix, $0; fflush(); }'
        else
            ssh "${SSH_OPTIONS[@]}" "${SSH_USER}@${host}" \
                "bash -lc $remote_command" >/dev/null 2>&1
        fi
    ) &
    pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
    host="${HOSTS[$index]}"
    prefix="[$host]"
    if [[ "$NODE_RANK_MODE" -eq 1 ]]; then
        prefix="[$host rank=${NODE_RANKS[$index]}]"
    fi
    if wait "${pids[$index]}"; then
        if [[ "$OUTPUT_ENABLED" -eq 1 ]]; then
            echo "$prefix command completed"
        fi
    else
        rc=$?
        echo "$prefix command failed with exit code $rc" >&2
        status=1
    fi
done

exit "$status"
