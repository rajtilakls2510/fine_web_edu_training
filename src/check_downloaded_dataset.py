from datatrove.pipeline.readers import ParquetReader
import time

if __name__ == "__main__":
    world_size = 20
    data_reader_rank = ParquetReader("./data/fine_web_data/data/CC-MAIN-2025-26")
    files_shard = data_reader_rank.data_folder.get_shard(0, world_size)
    print(f"Files shard: {files_shard}")
    start = time.perf_counter_ns()
    for i, doc in enumerate(data_reader_rank.read_files_shard(files_shard)):
        if i == 0:
            print(f"Document: {doc}")
            # print(f"Document: text {doc.text} token_count:{doc.metadata['token_count']}")
        end = time.perf_counter_ns()
        print(f"\r{i} docs read in {(end - start) / 1e6} ms", end="")

    