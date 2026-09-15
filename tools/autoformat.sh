#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

CHECK_ONLY=${CHECK_ONLY:-true}
SKIP_DOCS=${SKIP_DOCS:-false}
BASE_REF=${BASE_REF:-main}

files=()
declare -A seen=()

is_included_python_file() {
    local file=${1#./}
    [[ $file == *.py && -f $file ]] || return 1
    [[ /$file/ != */build/* ]] || return 1
    [[ /$file/ != */.venv/* ]] || return 1
    [[ /$file/ != */generated/* ]] || return 1
    [[ /$file/ != */data/* ]] || return 1
}

add_file() {
    local file=$1
    if is_included_python_file "$file" && [[ -z ${seen["$file"]+x} ]]; then
        files+=("$file")
        seen["$file"]=1
    fi
}

add_nul_delimited_files() {
    local file
    while IFS= read -r -d '' file; do
        add_file "$file"
    done
}

if (( $# > 0 )); then
    for file in "$@"; do
        add_file "$file"
    done
else
    if git show-ref --verify --quiet "refs/remotes/origin/$BASE_REF"; then
        comparison_ref="origin/$BASE_REF"
    elif git show-ref --verify --quiet "refs/heads/$BASE_REF"; then
        comparison_ref="$BASE_REF"
    else
        comparison_ref=HEAD
    fi

    add_nul_delimited_files < <(
        git diff --name-only -z --diff-filter=ACMR "$comparison_ref...HEAD" -- '*.py'
    )
    add_nul_delimited_files < <(
        git diff --cached --name-only -z --diff-filter=ACMR -- '*.py'
    )
    add_nul_delimited_files < <(
        git diff --name-only -z --diff-filter=ACMR -- '*.py'
    )
    add_nul_delimited_files < <(
        git ls-files --others --exclude-standard -z -- '*.py'
    )
fi

if (( ${#files[@]} == 0 )); then
    echo "No Python files to check."
    exit 0
fi

pylint_args=()
if [[ $SKIP_DOCS == true ]]; then
    pylint_args+=("--disable=C0115,C0116")
fi

if [[ $CHECK_ONLY == false ]]; then
    black "${files[@]}"
    isort "${files[@]}"
    ruff check --fix "${files[@]}"
    flake8 "${files[@]}"
    pylint "${pylint_args[@]}" "${files[@]}"
else
    black --check --diff "${files[@]}"
    isort --check-only "${files[@]}"
    flake8 "${files[@]}"
    pylint "${pylint_args[@]}" "${files[@]}"
    ruff check --no-fix "${files[@]}"
fi
