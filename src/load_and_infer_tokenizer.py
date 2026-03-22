from datatrove.pipeline.readers import ParquetReader
from tokenizers import Tokenizer
import torch

# =========================
# Config
# =========================
TOKENIZER_PATH = "./models/tokenizer.json"
DATA_PATH      = "./data/fine_web_data/data/CC-MAIN-2025-26"

BATCH_SIZE_TEXT = 4          # keep SMALL for debugging
CONTEXT_SIZE    = 32+1         # small so you can inspect chunks
WORLD_SIZE      = 20
RANK            = 0
MIN_TEXT_LENGTH = 20

PRINT_FIRST_N_BATCHES = 2
PRINT_FIRST_N_DOCS    = 2

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

# =========================
# Load tokenizer
# =========================
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

bos_id = tokenizer.token_to_id(BOS_TOKEN)
eos_id = tokenizer.token_to_id(EOS_TOKEN)

print("Special tokens:")
print(f"{BOS_TOKEN}: {bos_id}, {EOS_TOKEN}: {eos_id}")
print("=" * 80)

# =========================
# Reader
# =========================
reader = ParquetReader(DATA_PATH)
files_shard = reader.data_folder.get_shard(RANK, WORLD_SIZE)

# =========================
# Text batch iterator
# =========================
def text_batch_iterator():
    batch = []

    for doc in reader.read_files_shard(files_shard):
        text = doc.text

        if text is None or len(text.strip()) < MIN_TEXT_LENGTH:
            continue

        batch.append(text)

        if len(batch) == BATCH_SIZE_TEXT:
            yield batch
            batch = []

    if batch:
        yield batch

# =========================
# Token batch generator (with debug)
# =========================
def token_batch_generator():
    buffer = []
    batch_count = 0

    for text_batch in text_batch_iterator():
        print(f"\n\n===== TEXT BATCH {batch_count} =====")

        # -------------------------
        # Print raw text
        # -------------------------
        for i, text in enumerate(text_batch[:PRINT_FIRST_N_DOCS]):
            print(f"\n--- Raw Text [{i}] ---")
            print(text)  # truncate for readability

        # -------------------------
        # Encode batch
        # -------------------------
        encodings = tokenizer.encode_batch(text_batch)

        for i, enc in enumerate(encodings[:PRINT_FIRST_N_DOCS]):
            print(f"\n--- Encoding [{i}] ---")

            print("Tokens:")
            print(enc.tokens)

            print("Token IDs:")
            print(enc.ids)

            # Decode to verify round-trip
            decoded = tokenizer.decode(enc.ids)
            print("Decoded:")
            print(decoded)

            print("-" * 50)

        # -------------------------
        # Add to buffer
        # -------------------------
        for enc in encodings:
            ids = enc.ids

            if eos_id is not None:
                ids = ids + [eos_id]

            buffer.extend(ids)

        print(f"\nBuffer length after batch: {len(buffer)}")

        # -------------------------
        # Emit chunks
        # -------------------------
        while len(buffer) >= CONTEXT_SIZE:
            chunk = buffer[:CONTEXT_SIZE]
            buffer = buffer[CONTEXT_SIZE:]

            print(f"\n>>> CHUNK EMITTED (size={CONTEXT_SIZE})")

            print("Chunk IDs:")
            print(chunk)

            print("Chunk decoded:")
            print(tokenizer.decode(chunk))

            print("=" * 80)

            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)

            yield x, y

        batch_count += 1

        if batch_count >= PRINT_FIRST_N_BATCHES:
            print("\nStopping early (debug mode)")
            break

# =========================
# Run
# =========================
if __name__ == "__main__":
    gen = token_batch_generator()

    for i, (x, y) in enumerate(gen):
        print(f"\nFinal tensor shapes: x={x.shape}, y={y.shape}")

        if i == 2:
            break