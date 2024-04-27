#!/bin/sh
nvidia-smi -l 150 -f /home/usuaris/imatge/joan.manel.cardenas/age_predictions/log_gpu.log --query-gpu=timestamp,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv &
python3 /home/usuaris/imatge/joan.manel.cardenas/age_predictions/input_data.py
