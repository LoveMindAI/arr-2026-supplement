#!/usr/bin/env bash
# Runs on the pod: install vLLM once, then score every checkpoint in order; delete each checkpoint's weights after scoring.
set -uo pipefail
cd /workspace/expB
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /root/hf.token)"
if [ ! -f /workspace/INSTALLED ]; then
  python3 -m venv /workspace/venv && source /workspace/venv/bin/activate
  pip install -q --upgrade pip uv && uv pip install -q "vllm>=0.11.0" hf_transfer && touch /workspace/INSTALLED
fi
source /workspace/venv/bin/activate
python -c "import vllm; print('VLLM', vllm.__version__)"
for repo in $(cat LADDER.txt); do
  tag=$(echo $repo | tr '/' '_')
  [ -f out/$tag.DONE ] && continue
  echo "STAGE start $repo $(date +%s)"
  python score_ladder.py $repo out/$tag.jsonl > logs/$tag.log 2>&1 && touch out/$tag.DONE && echo "STAGE done $repo $(date +%s)" || echo "STAGE FAIL $repo"
  rm -rf $HF_HOME/hub/models--$(echo $repo | sed 's#/#--#')
done
echo ALL_DONE
