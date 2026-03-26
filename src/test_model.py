
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoModelForCausalLM
)
from tokenizers import Tokenizer
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_PATH = "./models/gpt2_base"
LORA_PATH = "./models/gpt2_lora"
MERGED_PATH = "./models/gpt2_merged"
TOKENIZER_PATH = "./models/tokenizer.json"
CONTEXT_SIZE = 1024

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 0. LOAD TOKENIZER
    # ============================================================
    print("\n[0] Loading tokenizer...")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    vocab_size = tokenizer.get_vocab_size()

    # ============================================================
    # 1. CREATE + SAVE BASE MODEL (GPT-2 from scratch)
    # ============================================================
    print("\n[1] Creating and saving base GPT-2 model...")

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=CONTEXT_SIZE,
        n_embd=256,
        n_layer=6,
        n_head=8,
        bos_token_id=tokenizer.token_to_id(BOS_TOKEN),
        eos_token_id=tokenizer.token_to_id(EOS_TOKEN),
        pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
        sliding_window=CONTEXT_SIZE
    )

    model = GPT2LMHeadModel(config).to(device)

    model.save_pretrained(MODEL_PATH)

    # ============================================================
    # 2. LOAD BASE MODEL
    # ============================================================
    print("\n[2] Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

    # ============================================================
    # 3. ATTACH LoRA
    # ============================================================
    print("\n[3] Attaching LoRA adapter...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # GPT-2 modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    # ============================================================
    # 4. SAVE LoRA ADAPTER
    # ============================================================
    print("\n[4] Saving LoRA adapter...")

    lora_model.save_pretrained(LORA_PATH)

    # ============================================================
    # 5. MERGE LoRA → FULL MODEL
    # ============================================================
    print("\n[5] Merging LoRA into base model...")

    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(MERGED_PATH)

    # ============================================================
    # 6. LOAD MERGED MODEL
    # ============================================================
    print("\n[6] Loading merged model...")

    model = AutoModelForCausalLM.from_pretrained(MERGED_PATH).to(device)

    # ============================================================
    # 7. FORWARD PASS (NO KV CACHE)
    # ============================================================
    print("\n[7] Forward pass (no KV cache)...")

    input_ids = torch.randint(0, vocab_size, (2, 16)).to(device)

    outputs = model(input_ids=input_ids)
    print("Logits shape:", outputs.logits.shape)

    # ============================================================
    # 8. FORWARD PASS (WITH KV CACHE)
    # ============================================================
    print("\n[8] Generation with KV cache...")

    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, 1)).to(device)
    past = None

    for _ in range(10):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True
        )

        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)

        past = outputs.past_key_values
        input_ids = next_token

        print(next_token.item(), end=" ", flush=True)
        print(f"{past =}")

    print("\n")

    # ============================================================
    # 9. LOAD BASE + ATTACH LoRA AGAIN (ALT WORKFLOW)
    # ============================================================
    print("\n[9] Reload base + attach LoRA...")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH).to(device)

    # Quick sanity check
    input_ids = torch.randint(0, vocab_size, (1, 8)).to(device)
    outputs = lora_model(input_ids=input_ids)

    print("LoRA model logits shape:", outputs.logits.shape)

    print("\n✅ All workflows completed successfully!")