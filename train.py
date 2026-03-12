import torch
import deepspeed

from torch.utils.data import DataLoader
from data.dataset import IMDBDataset
from models.model import build_model
from utils.gpu_monitor import GPUMonitor


def main():

    dataset = IMDBDataset("train")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = build_model()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config="configs/ds_config.json"
    )

    gpu_monitor = GPUMonitor()
    global_step = 0

    for epoch in range(3):

        for idx,batch in enumerate(dataloader):

            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            outputs = model_engine(**batch)

            loss = outputs.loss

            model_engine.backward(loss)

            model_engine.step()

            if global_step%100 == 0:
                print(f"epoch {epoch} | batch {idx} | loss {loss.item():.4f}")
                gpu_monitor.record()
            global_step += 1

        if model_engine.global_rank == 0:
            save_dir = f"checkpoints/epoch_{epoch}"
            model_engine.save_checkpoint(save_dir)
            print(f"checkpoint saved: {save_dir}")

        print("epoch finished")


if __name__ == "__main__":
    main()