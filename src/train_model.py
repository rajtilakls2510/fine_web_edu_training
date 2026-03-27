import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer
from accelerate import Accelerator
from dataset import FineWebIterableDataset

MODEL_PATH = "./models/my_new_model"
CHECKPOINT_DIR = "./checkpoints"

TOKENIZER_PATH = "./models/tokenizer.json"
DATA_PATH = "./data/fine_web_data/data/CC-MAIN-2025-26"

BATCH_SIZE = 16
CONTEXT_SIZE = 512
GRAD_ACCUM_STEPS = 256
NUM_WORKERS = 10
LR = 1e-4
MAX_GRAD_NORM = 1.0

SAVE_EVERY = 1000   # steps

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

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        # mixed_precision="bf16"   # optional
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
    for step, batch in enumerate(loader, start=global_step):

        with accelerator.accumulate(model):

            outputs = model(
                input_ids=batch[:, :-1],
                labels=batch[:, 1:]
            )

            loss = outputs.loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()

        # ========================================================
        # LOGGING
        # ========================================================
        if step % 10 == 0:
            accelerator.print(f"Step {step} | Loss: {loss.item():.4f}")

        # ========================================================
        # SAVE CHECKPOINT
        # ========================================================
        if step % SAVE_EVERY == 0 and accelerator.is_main_process:

            ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}")
            os.makedirs(ckpt_path, exist_ok=True)

            accelerator.print(f"Saving checkpoint at step {step}")

            accelerator.save_state(ckpt_path)

    # ============================================================
    # FINAL SAVE
    # ============================================================
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(MODEL_PATH)

    accelerator.print("\n✅ Training complete!")


if __name__ == "__main__":
    main()