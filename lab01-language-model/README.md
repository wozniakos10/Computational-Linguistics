In this laboratory the goal was to train from scratch a language model that generates sentences in Polish language with 2 architectures:

- RNN
- Transformer (decoder only)

For the transformer, a GPT-2 like architecture was used and for the RNN, an architecture based on LSTM.
Polish language data was grabbed from [speakleash](https://github.com/speakleash/speakleash)

## Content
There is a simple guide through important files:

- `dataset.py` - PyTorch dataloader for speakleash data
- `transformer_based_llm.py` - GPT-2 like architecture created in PyTorch
- `rnn_based_llm.py` - RNN architecture created in PyTorch
- `models.py` - Pydantic models for storing models, dataset, and training parameters
- `utils.py` - Helpful logic like encoding/decoding, loss calculation, etc.
- `llm_train.py` - Script to run with particular configuration. Most parameters have to be manually changed in that script by editing 
- `transformer_inference.py` - Script to test transformer inference efficiency and output quality
- `rnn_inference.py` - Script to test rnn inference efficiency and output quality


`MODEL_CONFIG`, `TRAINING_SETTINGS` or `DATASET_SETTINGS`. A couple of parameters and logic are handled by `argparse`.
Example execution:
```
python llm_train.py --model_type transformer --tokenizer flax-community/papuGaPT2 --dataset_name wolne_lektury_corpus --max_docs 1000000 --use_tiktoken false
```
By that, the following can be customized:
- model_type: transformer or rnn
- tokenizer and use_tiktoken: Tokenizer name and information if it is available in tiktoken. If use_tiktoken is set to false, `AutoTokenizer` from `Transformers` will be used.
- dataset_name: Name of dataset from speakleash
- max_docs: Max docs to use from dataset. If max_docs > amount of documents in dataset - all documents will be used

Report containing deeper information and results can be found in [report.md](report.md)