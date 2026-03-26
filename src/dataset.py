import torch
from datatrove.pipeline.readers import ParquetReader
from tokenizers import Tokenizer
import time

TOKENIZER_PATH = "./models/tokenizer.json"
DATA_PATH      = "./data/fine_web_data/data/CC-MAIN-2025-26"

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

CONTEXT_SIZE = 512+1

class FineWebIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_path: str = None, tokenizer_path: str = None, context_size = 1024):
        super(FineWebIterableDataset).__init__()
        assert data_path is not None, "'data_path' must be provided"
        assert tokenizer_path is not None, "'tokenizer_path' must be provided"
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.context_size = context_size

    def __iter__(self):
        self.reader = ParquetReader(self.data_path)
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        # Cache special token ids
        self.bos_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # Single process data loading
            files_shard = self.reader.data_folder.get_shard(0, 1)   # Returns all the files
        else:   # Multi-process data loading
            files_shard = self.reader.data_folder.get_shard(worker_info.id, worker_info.num_workers)
            print(f"Worker: {worker_info.id}/{worker_info.num_workers} Files Shard: {files_shard}")
        buffer = []
        buffer.append(self.bos_id)
        read_iterator = iter(self.reader.read_files_shard(files_shard))

        while len(buffer) < self.context_size:
            try:
                text = next(read_iterator).text
                if text is None:
                    continue
            except StopIteration:
                break

            encoding = self.tokenizer.encode(text)
            tokens = encoding.ids
            tokens = tokens + [self.eos_id]
            
            idx = 0
            while idx < len(tokens):
                remaining_space = self.context_size - len(buffer)
                take = min(remaining_space, len(tokens) - idx)

                buffer.extend(tokens[idx: idx + take])
                idx += take

                if len(buffer) == self.context_size:
                    yield torch.tensor(buffer, dtype=torch.long)
                    buffer = [self.bos_id]   # Start next sequence with BOS
        if len(buffer) > 1:
            # pad remaining
            buffer = buffer + [self.pad_id] * (self.context_size - len(buffer))
            yield torch.tensor(buffer, dtype=torch.long)

if __name__ == "__main__":
    ds = FineWebIterableDataset(data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH, context_size=CONTEXT_SIZE)
    
    start_time = time.time()
    num_batches = 0
    for batch_idx, batch in enumerate(
        torch.utils.data.DataLoader(ds, num_workers=20, batch_size=1024)
    ):
        num_batches += 1

        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            bps = num_batches / elapsed
            print(f"\r{batch_idx=} {batch.shape=} {bps:.2f} batches/sec", end="", flush=True)


