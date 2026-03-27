import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from dataset import FineWebIterableDataset
import time

MODEL_PATH = "./models/my_new_model"
CHECKPOINT_DIR = "./checkpoints"

TOKENIZER_PATH = "./models/tokenizer.json"
DATA_PATH = "./data/fine_web_data/data/CC-MAIN-2025-26"

BATCH_SIZE = 16
CONTEXT_SIZE = 512
GRAD_ACCUM_STEPS = 256
NUM_WORKERS = 10
LR = 1e-3
MAX_GRAD_NORM = 1.0

SAVE_EVERY = 1024   # steps

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"


def get_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None

    checkpoints = [
        d for d in os.listdir(CHECKPOINT_DIR)
        if d.startswith("step_")
    ]

    if len(checkpoints) == 0:
        return None

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))
    return os.path.join(CHECKPOINT_DIR, checkpoints[-1])


def main():
    print("Starting training...")

    project_config = ProjectConfiguration(project_dir=".", logging_dir="logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        log_with="tensorboard",
        project_config=project_config
        # mixed_precision="bf16"   # optional
    )

    accelerator.init_trackers(
        project_name="gpt2-from-scratch",
        config={
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "context_size": CONTEXT_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
        },
    )

    # ============================================================
    # DATA
    # ============================================================
    ds = FineWebIterableDataset(
        data_path=DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        context_size=CONTEXT_SIZE + 1
    )

    loader = torch.utils.data.DataLoader(
        ds,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    # ============================================================
    # MODEL
    # ============================================================
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    config = GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=CONTEXT_SIZE,
        n_embd=256,
        n_layer=6,
        n_head=8,
        bos_token_id=tokenizer.token_to_id(BOS_TOKEN),
        eos_token_id=tokenizer.token_to_id(EOS_TOKEN),
        pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
    )

    model = GPT2LMHeadModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Total params: {total_params:,}")
    accelerator.print(f"Trainable params: {trainable_params:,}")

    # Optional memory optimization
    model.gradient_checkpointing_enable()

    # ============================================================
    # OPTIMIZER + SCHEDULER
    # ============================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ============================================================
    # PREPARE
    # ============================================================
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    model.train()

    # ============================================================
    # RESUME FROM CHECKPOINT
    # ============================================================
    global_step = 0

    latest_ckpt = get_latest_checkpoint()

    if latest_ckpt is not None:
        accelerator.print(f"Resuming from {latest_ckpt}")
        accelerator.load_state(latest_ckpt)

        global_step = int(latest_ckpt.split("_")[-1])

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    start_time = time.time()
    grad_norm = None
    for step, batch in enumerate(loader, start=global_step+1):

        with accelerator.accumulate(model):

            outputs = model(
                input_ids=batch[:, :-1],
                labels=batch[:, 1:]
            )

            loss = outputs.loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()

        # ========================================================
        # LOGGING
        # ========================================================
        if step % 10 == 0:
            accelerator.print(f"Step {step} | Loss: {loss.item():.4f}")
            
            elapsed = time.time() - start_time
            tokens = BATCH_SIZE * CONTEXT_SIZE * GRAD_ACCUM_STEPS
            tokens_per_sec = tokens / elapsed

            accelerator.log(
                {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                    "tokens/sec": tokens_per_sec,
                },
                step=step
            )

            start_time = time.time()

        # ========================================================
        # SAVE CHECKPOINT
        # ========================================================
        if step % SAVE_EVERY == 0 and accelerator.is_main_process:

            ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}")
            os.makedirs(ckpt_path, exist_ok=True)

            accelerator.print(f"Saving checkpoint at step {step}")

            accelerator.save_state(ckpt_path)

            # Inference checkpoint
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(MODEL_PATH)

    # ============================================================
    # FINAL SAVE
    # ============================================================
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(MODEL_PATH)

    accelerator.print("\n✅ Training complete!")


if __name__ == "__main__":
    main()