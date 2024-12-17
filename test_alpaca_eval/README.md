# Alpaca Eval 2.0

Source: [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)

## Env Setup

```bash
pip install alpaca-eval
```

## Preprocess: Transform to Alpaca Format

```bash
python eval/alpaca_eval/preprocess_alpaca_eval.py \
    --output_file data/alpaca_eval.json
# Transformed 805 responses to data/alpaca_eval.json in alpaca format
```

Add the following to `data/dataset_info.json`:

```json
  "alpaca_eval": {
    "file_name": "alpaca_eval.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
```

## Run Eval

```bash
export OPENAI_API_KEY=<your_api_key>
MODEL_NAME_OR_PATH=saves/sft/conifer/conifer_mistral_7b_v0.1/sft_full
TEMPLATE=mistral
OUTPUT_DIR=saves/eval/sft/conifer/conifer_mistral_7b_v0.1/sft_full/alpaca_eval
EVAL_BATCH_SIZE=32
GPT_ANNOTATOR=weighted_alpaca_eval_gpt4_turbo
# "weighted_alpaca_eval_gpt4_turbo" by default, which points to "gpt-4-1106-preview" at 2024.11;
# use "weighted_alpaca_eval_gpt-4o-mini-2024-07-18" for lower price
# Run on 1 GPU, takes 0.5~1h to generate responses, cost ~8USD
bash eval/alpaca_eval/run_alpaca_eval.sh $MODEL_NAME_OR_PATH $TEMPLATE $OUTPUT_DIR $EVAL_BATCH_SIZE $GPT_ANNOTATOR
```