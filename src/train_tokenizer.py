from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from datatrove.pipeline.readers import ParquetReader
import time

DATA_PATH = "./data/fine_web_data/data/CC-MAIN-2025-26"
TOKENIZER_PATH = "./models/tokenizer.json"

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

VOCAB_SIZE = 32000

tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN])

def batch_iterator(reader, files_shard, batch_size=1000, max_docs=None):
    batch = []
    count = 0

    for doc in reader.read_files_shard(files_shard):
        if doc.text is None or len(doc.text.strip()) == 0:
            continue

        batch.append(doc.text)
        count += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

        if max_docs and count >= max_docs:
            break

    if batch:
        yield batch



if __name__ == "__main__":
    world_size = 20
    data_reader = ParquetReader(DATA_PATH)

    start_time = time.perf_counter_ns()

    for rank in range(world_size):
        print(f"Rank: {rank}")

        shard_start_time = time.perf_counter_ns()
        files_shard = data_reader.data_folder.get_shard(rank, world_size)

        tokenizer.train_from_iterator(
            batch_iterator(data_reader, files_shard),
            trainer=trainer,
            length=None   # optional (can estimate if you want)
        )
        tokenizer.save(TOKENIZER_PATH)
        shard_end_time = time.perf_counter_ns()
        print(f"Shard Processing Time: {(shard_end_time - shard_start_time) / 1e9} s")
    print(f"Total Processing Time: {(shard_end_time - start_time) / 1e9} s")
    