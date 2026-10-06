#!/usr/bin/env bash
# Checks that what is running is this config and not a similar one: active patches, the KV cache,
# a real generation and the speed. The patches fail silently (it just gets slower).
set -uo pipefail
cd "$(dirname "$0")/.."
KEY=$(grep -oP '^API_KEY=\K.*' .env); PORT=$(grep -oP '^PORT=\K.*' .env || echo 8010)
ok(){ echo -e "  \033[32m✓\033[0m $*"; }; bad(){ echo -e "  \033[31m✗\033[0m $*"; F=1; }
F=0

echo "== waiting for /health (the first startup takes 10-30 min)"
r0=$(docker inspect -f '{{.RestartCount}}' vllm-fn)
until curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null; do
  # A failed startup does NOT stop the container: with restart unless-stopped it loops
  # and `docker ps` still says Up. RestartCount is what tells.
  if [ "$(docker inspect -f '{{.RestartCount}}' vllm-fn)" != "$r0" ]; then
    bad "crash loop — last lines:"; docker logs --tail 40 vllm-fn 2>&1 | grep -iE "error|Traceback|memory" | tail -8; exit 1
  fi
  sleep 15
done
ok "serving on :$PORT"

echo "== patches"
L=$(docker logs vllm-fn 2>&1 | grep -v APIServer)
n=$(grep -oP 'fn_dense_fp8: \K[0-9]+' <<<"$L" | tail -1)
[ "$n" = 145 ] && ok "dense FP8: 145 layers" || bad "dense FP8: '${n:-not active}' (expected 145; is config/fn_dense_fp8.conf missing?)"
grep -q "built NVFP4 draft head" <<<"$L" && ok "NVFP4 draft head" || bad "NVFP4 draft head not found in the logs"
grep -q "RecoverSSM speculative verify active" <<<"$L" && ok "RecoverSSM verify active" || bad "RecoverSSM not active (use-replayssm / VLLM_USE_V2_MODEL_RUNNER?)"
bs=$(grep -oP 'Setting attention block size to \K[0-9]+' <<<"$L" | tail -1)
[ "$bs" = 1696 ] && ok "attention block 1696 (prefix retention 1696)" || bad "attention block '${bs:-?}' (expected 1696; retention must be a multiple of it)"
grep -oP 'GPU KV cache size: [0-9,]+ tokens' <<<"$L" | tail -1 | sed 's/^/  · /;s/$/  (DGX Spark reference: ~294-309k with embed+whisper running)/'

echo "== generation (official sampling; at temp 0 it loops)"
KEY=$KEY PORT=$PORT python3 - <<'P' || F=1
import json, os, time, urllib.request
def ask(prompt, n):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{os.environ['PORT']}/v1/chat/completions",
        json.dumps(body).encode(), {"Authorization": f"Bearer {os.environ['KEY']}", "Content-Type": "application/json"})
    t = time.time(); j = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t
    return j["choices"][0]["message"]["content"], j["usage"]["completion_tokens"], dt
ask("Hello", 8)  # warm-up
txt, n, dt = ask("Write a Python LRUCache class with O(1) get and put, with docstrings and tests.", 512)
print(f"  · {n} tokens in {dt:.1f} s = {n/dt:.1f} t/s  (DGX Spark reference, code: ~54-58 t/s; includes prefill)")
print("  · " + txt.strip().splitlines()[0][:100])
P
[ "$F" = 0 ] && ok "verified" || { bad "see failures above"; exit 1; }
