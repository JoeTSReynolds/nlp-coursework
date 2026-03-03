import os
import time
import subprocess
from datetime import datetime
import logging

# --- ENVIRONMENT TOGGLE ---
# Set to True when running via Slurm on gpucluster
# Set to False when running interactively in tmux on a lab machine
RUN_ON_CLUSTER = True

def setup_logger(log_filename="training_run.log"):
    logger = logging.getLogger("PCL_Logger")
    logger.setLevel(logging.INFO)
    
    # prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # prints to file
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # prints to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def smart_vram_guard(min_gb_required=7.5, check_interval=60, logger=None):
    """
    Bypasses the VRAM check if on the cluster (since Slurm handles allocation).
    If on a lab machine, waits for free VRAM.
    """
    if RUN_ON_CLUSTER:
        if logger:
            logger.info("RUN_ON_CLUSTER=True: Bypassing VRAM guard. Slurm has allocated a dedicated GPU.")
        return

    msg = f"VRAM Guard: Checking for {min_gb_required}GB of free space on Lab Machine..."
    if logger: 
        logger.info(msg)
    else: 
        print(msg)

    while True:
        try:
            res = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader'
            ])
            free_mem_mb = int(res.decode().strip().split('\n')[0])
            free_gb = free_mem_mb / 1024

            if free_gb >= min_gb_required:
                success_msg = f"Success: {free_gb:.2f}GB free. Starting training now."
                if logger: 
                    logger.info(success_msg)
                else: 
                    print(success_msg)
                break
            else:
                wait_msg = f"Only {free_gb:.2f}GB free. Waiting {check_interval}s..."
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {wait_msg}", end='\r')
                time.sleep(check_interval)
        except Exception as e:
            err_msg = f"Error checking VRAM: {e}. Retrying in 10s..."
            if logger:
                logger.error(err_msg)
            else: 
                print(err_msg)
            time.sleep(10)