#!/usr/bin/env bash
# Score every IPAD test sequence with one checkpoint.
#
#   scripts/eval_sweep.sh configs/ipad.yaml /path/to/checkpoint.pt /path/to/out_dir
#
# Writes one <name>.txt of per-frame anomaly scores per sequence, plus a run.log
# summarising each invocation. Replaces the six near-identical run_w_mse_*.sh scripts;
# the checkpoint and output directory are now arguments rather than edited in place.
#
# Sequences default to the six used in the paper. Override with SEQUENCES, giving
# "name=relative/path.npy" entries resolved against the dataset's data root:
#
#   SEQUENCES="type_1_006=mod_image/test/type_1/npy/006_mod.npy" scripts/eval_sweep.sh ...

set -uo pipefail

CONFIG="${1:?usage: $0 <config.yaml> <checkpoint.pt> <out_dir>}"
CKPT="${2:?usage: $0 <config.yaml> <checkpoint.pt> <out_dir>}"
OUT_DIR="${3:?usage: $0 <config.yaml> <checkpoint.pt> <out_dir>}"

DEFAULT_SEQUENCES="
type_1_006=mod_image/test/type_1/npy/006_mod.npy
type_1_007=mod_image/test/type_1/npy/007_mod.npy
type_1_008=mod_image/test/type_1/npy/008_mod.npy
type_2_005=mod_image/test/type_2/npy/005_mod.npy
type_2_013=mod_image/test/type_2/npy/013_mod.npy
type_2_014=mod_image/test/type_2/npy/014_mod.npy
"
SEQUENCES="${SEQUENCES:-$DEFAULT_SEQUENCES}"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run.log"
: > "${LOG}"

# Resolve the data root the same way the Python code does, so relative sequence paths
# mean the same thing here as they do in the config.
DATA_ROOT="$(python -m sparsest.train --config "${CONFIG}" --print-config \
  | python -c 'import json,sys; print(json.load(sys.stdin)["data"]["root"])')"

echo "[$(date '+%F %T')] sweep start  ckpt=${CKPT}  data_root=${DATA_ROOT}" >> "${LOG}"

status_overall=0
for entry in ${SEQUENCES}; do
  name="${entry%%=*}"
  rel="${entry#*=}"
  input="${rel}"
  [[ "${input}" = /* ]] || input="${DATA_ROOT}/${rel}"
  output="${OUT_DIR}/${name}.txt"

  echo "[$(date '+%F %T')] running ${name} <- ${input}" >> "${LOG}"

  SPARSEST_TEST_NPY="${input}" python -m sparsest.evaluate \
    --config "${CONFIG}" \
    --eval-mode anomaly_score \
    --checkpoint "${CKPT}" \
    -batch_size 1 \
    > "${output}" 2>&1
  status=$?

  scores=$(grep -c '^anomaly score for image' "${output}" 2>/dev/null || echo 0)
  echo "[$(date '+%F %T')] finished ${name} status=${status} scores=${scores}" >> "${LOG}"
  [[ ${status} -ne 0 ]] && status_overall=${status}
done

echo "[$(date '+%F %T')] sweep done status=${status_overall}" >> "${LOG}"
echo "wrote ${OUT_DIR} (see ${LOG})"
exit "${status_overall}"
