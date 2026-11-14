#!/bin/bash
# Flash Next on vLLM v0.33.0: overlay -> guard1 -> i3pl -> yarn58879 -> gb10r4. Usage: build-v033.sh [from-step]; steps: overlay guard i3pl yarn gb10
set -euo pipefail
S=~/build/v033-src/images
O=~/third_party/qwen3.8-Flash-DGX-v033
B=qwen38-flash-dgx:v0.33.0-blazux5108d90r
FROM=${1:-overlay}
order="overlay guard i3pl yarn gb10 wfix1 wfix2"; run=0
for st in $order; do
  [ "$st" = "$FROM" ] && run=1
  [ $run = 1 ] || continue
  echo "=== STEP $st $(date +%T)"
  case $st in
    overlay)
      rm -rf $O && cp -a ~/third_party/qwen3.8-Flash-DGX-v031 $O
      cp $S/overlays/qwen3.8-Flash-DGX-v033/Dockerfile.v0.33 $O/
      cp $S/overlays/qwen3.8-Flash-DGX-v033/src/*.py $O/src/; cp $S/overlays/qwen3.8-Flash-DGX-v033/src/patches/* $O/src/patches/
      mkdir -p $O/src/kernel-det && cp -a ~/build/kernel-det-stage/. $O/src/kernel-det/
      cd $O && docker build --progress=plain -f Dockerfile.v0.33 -t $B . ;;
    guard) cd $S/guard && docker build --progress=plain -f Dockerfile.guard --build-arg BASE_IMAGE=$B -t $B-guard1 . ;;
    i3pl) cd $S/patches/think-eos-guard-i3pl && docker build --progress=plain --build-arg BASE=$B-guard1 --build-arg DIFF=serving_i3pl_v033.diff -t $B-guard1-i3pl . ;;
    yarn) cd $S/patches/yarn-mrope-58879 && docker build --progress=plain --build-arg BASE=$B-guard1-i3pl --build-arg DIFF=yarn_mrope_58879_v032.diff -t $B-guard1-i3pl-yarn58879 . ;;
    gb10) cd $S/patches/gb10-decode && docker build --progress=plain --build-arg BASE=$B-guard1-i3pl-yarn58879 -t $B-guard1-i3pl-yarn58879-gb10r4 . ;;
    wfix1) cd $S/patches/cutedsl-warmup-v032 && docker build --progress=plain --build-arg BASE=$B-guard1-i3pl-yarn58879-gb10r4 -t $B-guard1-i3pl-yarn58879-gb10r4-wfix1 . ;;
    wfix2) cd $S/patches/mtp-rope-v032 && docker build --progress=plain --build-arg SCRIPT=apply_mtp_rope_v033.py --build-arg BASE=$B-guard1-i3pl-yarn58879-gb10r4-wfix1 -t $B-guard1-i3pl-yarn58879-gb10r4-wfix2 . ;;
  esac
done
echo BUILD-DONE $(date +%T)
