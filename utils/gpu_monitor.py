import torch
import csv
import os

class GPUMonitor:

    def __init__(self, log_file="logs/memory_log.csv"):

        self.log_file = log_file
        self.step = 0

        os.makedirs("logs", exist_ok=True)

        with open(self.log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["batch", "gpu_memory_MB"])

    def record(self):

        memory = torch.cuda.memory_allocated() / 1024 / 1024

        with open(self.log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.step, memory])

        self.step += 100