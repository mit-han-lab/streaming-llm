# Efficient Streaming Language Models with Attention Sinks [[paper](http://arxiv.org/abs/2309.17453)]

![schemes](figures/schemes.png)


https://github.com/mit-han-lab/streaming-llm/assets/40906949/f0af8b59-d91a-4395-bf14-1bd9acbdfa87


## TL;DR
We deploy LLMs for infinite-length inputs without sacrificing efficiency and performance.

## Abstract
Deploying Large Language Models (LLMs) in streaming applications such as multi-round dialogue, where long interactions are expected, is urgently needed but poses two major challenges. Firstly, during the decoding stage, caching previous tokens' Key and Value states (KV) consumes extensive memory. Secondly, popular LLMs cannot generalize to longer texts than the training sequence length. Window attention, where only the most recent KVs are cached, is a natural approach --- but we show that it fails when the text length surpasses the cache size. We observe an interesting phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover the performance of window attention. In this paper, we first demonstrate that the emergence of attention sink is due to the strong attention scores towards initial tokens as a ``sink'' even if they are not semantically important. Based on the above analysis, we introduce StreamingLLM, an efficient framework that enables LLMs trained with a finite length attention window to generalize to infinite sequence length without any fine-tuning. We show that StreamingLLM can enable Llama-2, MPT, Falcon, and Pythia to perform stable and efficient language modeling with up to 4 million tokens and more. In addition, we discover that adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment. In streaming settings, StreamingLLM outperforms the sliding window recomputation baseline by up to 22.2x speedup.

## Usage

### Environment Setup

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers accelerate datasets evaluate wandb scikit-learn scipy

python setup.py develop
```

### Run Streaming Llama Chatbot

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py  --enable_streaming
```

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release core code of StreamingLLM, including Llama-2, MPT, Falcon, and Pythia.
- [x] Release perplexity evn code
- [x] Release Streaming Llama Chatbot demo.
- [ ] Release StreamEval dataset and evaluation code.

## Citation

If you find StreamingLLM useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{xiao2023streamingllm,
        title={Efficient Streaming Language Models with Attention Sinks},
        author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
        journal={arXiv},
        year={2023}
        }
```
