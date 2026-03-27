from transformers import PreTrainedTokenizerFast

MODEL_REPO_NAME = "rajtilakls2510/gpt2-pretraining"
TOKENIZER_PATH = "./models/tokenizer.json"

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
    )

    tokenizer.push_to_hub(MODEL_REPO_NAME)