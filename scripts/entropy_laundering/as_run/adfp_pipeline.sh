#!/usr/bin/env bash
# ADFP lambda-256 masking study. Usage: adfp_pipeline.sh GEN_PID
set -uo pipefail
cd "$(dirname "$0")"
gen_pid=$1
A=$HOME/textseal-evals/adfp
TR=$A/traces/lambda256/traces.jsonl
log() { echo "$* $(date -u +%FT%TZ)" >> logs/stages.log; }
while kill -0 $gen_pid 2>/dev/null; do sleep 120; done
n=$(wc -l < $TR 2>/dev/null || echo 0)
[ "$n" -ge 9000 ] || { log "adfp256 generation ended with only $n traces; stopping"; exit 1; }
log "adfp256 traces ready ($n)"
# lambda-64 generation on GPUs 6,7 in the background.
( cd $A && R=/ephemeral/codex-textseal && mkdir -p traces/lambda64 && \
  HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$R/length-repo CUDA_VISIBLE_DEVICES=6,7 \
  SCIENCE_TRAIN_PATH=$R/length-repo/data/science/train_9k.jsonl SCIENCE_EVAL_PATH=$R/length-repo/data/science/eval_1k.jsonl \
  TRITON_CACHE_DIR=$A/cache/triton67 TORCHINDUCTOR_CACHE_DIR=$A/cache/inductor67 RUNTIME_CACHE_ROOT=$A/cache/runtime67 \
  $R/venv/bin/python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port=29941 \
    --nproc_per_node=2 $R/length-repo/stages/stage1_generate.py --dataset science --split train \
    --max-examples 9000 --teacher-model Qwen/Qwen3.5-9B --teacher-adapter $R/reproduction/teacher \
    --teacher-dtype bfloat16 --proxy-model Qwen/Qwen3.5-4B --teacher-pad-token '' --proxy-pad-token '' \
    --method ads --lam 64 --hash-config $R/reproduction/hash.json \
    --output traces/lambda64/traces.jsonl --metadata traces/lambda64/metadata.json --batch-size 24 --seed 42 \
    --max-new-tokens 3840 --temperature 0.7 --top-p 0.95 --repetition-penalty 1.0 --length-hints off \
    > logs/generate-lambda64.log 2>&1; echo "lambda64 generate done (exit $?) $(date -u +%FT%TZ)" >> logs/stages.log ) &
# Detection subset (first 3,000 traces) and Qwen-reference entropy (short job, GPU 0).
head -n 3000 $TR > $A/traces/lambda256/detect-3000.jsonl
./run_entropy.sh 0 $TR entropy/adfp256.npz > logs/entropy-adfp256.log 2>&1 || log "adfp256 entropy FAILED"
log "adfp256 entropy done"
./run_detect_adfp.sh 1 control-adfp256 /ephemeral/codex-textseal/length4096/control-student $A/traces/lambda256/detect-3000.jsonl open &
# Students on GPUs 4,5; detection + eval of each on GPUs 0-3 as it finishes.
jobs=("adfp256-drop0:0.0:" "adfp256-top-entropy-50:0.5:--mask-mode top-entropy --entropy-file entropy/adfp256.npz"
      "adfp256-top-surprisal-50:0.5:--mask-mode top-surprisal --entropy-file entropy/adfp256.npz"
      "adfp256-drop50:0.5:" "adfp256-bottom-entropy-50:0.5:--mask-mode bottom-entropy --entropy-file entropy/adfp256.npz")
g=0
for job in "${jobs[@]}"; do
  IFS=: read -r name rate extra <<< "$job"
  EXTRA_ARGS="$extra" ./run_drop.sh 4,5 29951 "$name:$TR:$rate"
  d=$((g % 2 * 2)); e=$((d + 1)); g=$((g + 1))
  ( ./run_detect_adfp.sh $d $name students/$name $A/traces/lambda256/detect-3000.jsonl open
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_BATCH=12 ./run_eval.sh $e $name ) &
done
wait
log "adfp256 pipeline done"
