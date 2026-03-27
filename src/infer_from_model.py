import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

MODEL_PATH = "./models/my_new_model"   # or your base model
TOKENIZER_PATH = "./models/tokenizer.json"

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

MAX_NEW_TOKENS = 50


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 1. LOAD TOKENIZER (HF wrapper)
    # ============================================================
    print("\n[1] Loading tokenizer...")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN
    )

    # ============================================================
    # 2. LOAD MODEL
    # ============================================================
    print("\n[2] Loading model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # ============================================================
    # 3. PREPARE INPUT
    # ============================================================
    prompt = "Hello, how are you"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"Prompt: {prompt}")
    print(f"Input IDs: {input_ids}")

    # ============================================================
    # 4. GENERATION WITH KV CACHE
    # ============================================================
    print("\n[3] Generating with KV cache...\n")

    past = None
    generated = input_ids

    for step in range(MAX_NEW_TOKENS):

        outputs = model(
            input_ids=generated[:, -1:] if past is not None else generated,
            past_key_values=past,
            use_cache=True
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # append
        generated = torch.cat([generated, next_token], dim=1)

        past = outputs.past_key_values

        # stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # ============================================================
    # 5. DECODE OUTPUT
    # ============================================================
    output_text = tokenizer.decode(generated[0].tolist())

    print("Generated text:\n")
    print(output_text)


if __name__ == "__main__":
    main()