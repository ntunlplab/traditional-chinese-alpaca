# Traditional-Chinese Alpaca Dataset

This repo aims to share resources for building Traditional-Chinese instruction-following language models (for research purposes only). This repo contains:
  - A Traditional-Chinese version of the Alpaca dataset -> [```alpaca_data-tw.json```](alpaca_data-tw.json)
  - Code for training and inferencing a Traditional-Chinese Alpaca-Lora LLaMA model. (to be added soon)
  
## About the dataset
We translate the [Stanford Alpaca 52k dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) directly to Traditional Chinese via the [ChatGPT API](https://platform.openai.com/docs/guides/chat) (```gpt-3.5-turbo```), which cost us roughly 40 USD.

## Fine-tuning
Ongoing, to be added soon.

## Next
1. Fine-tune various multi-lingual foundation models (e.g., [bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)).
2. Construct a large-scale Traditional-Chinese instruction-following dataset. 
2. Construct domain-specific Traditional-Chinese instruction-following datasets.

*Please feel free to reach out (contact[at]nlg.csie.ntu.edu.tw) if you are interested in any forms of collaborations!*

## Reference
A large portion of our work relies on/motivated by [LLaMA](https://arxiv.org/abs/2302.13971), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), [ChatGPT](https://openai.com/blog/chatgpt), [Hugging Face](https://huggingface.co/), and [Cabrita](https://github.com/22-hours/cabrita).
We thanks the incredible individuals, groups, and communities for opening their amazing works!

## Citation
If you use the data or code from this repo, please cite this repo as follows
```
@misc{traditional-chinese-alpaca,
  author = {Wei-Lin Chen and Cheng-Kuang Wu and Hsin-Hsi Chen},
  title = {A Traditional Chinese Version of the Alpaca Dataset for Instruction-Finetuning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ntunlplab/traditional-chinese-alpaca}},
}
```
