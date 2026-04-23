#!/usr/bin/env bash

srun --workspace-name expert-services \
  -p a800 \
  -f pt \
  -r N3lS.Ii.I60.1 \
  --container-image "$SCC_IMAGE" \
  -j test_new_audio_reader \
  -o /home/mnt/cuishaoting/LazyLLM/try_audio/test_new_audio_reader.log \
  "cd /home/mnt/cuishaoting/LazyLLM && python /home/mnt/cuishaoting/LazyLLM/try_audio/test_new_audio_reader.py"
