# TSV Guided Hallucination Detection with ITI Approach

This project is an extension based on 1) the ICML 2025 paper: [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917) by Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li, and 2) the NeurIPS 2023 paper: [Inference-Time Intervention:
Eliciting Truthful Answers from a Language Model] (https://arxiv.org/abs/2306.03341) by Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister and Martin Wattenberg.

The majority of the code were adapted from the TSV source code of the paper mentioned above. See the source code here: (https://github.com/deeplearning-wisc/tsv.git)

---

# Our innovation
The curren project is an extension of the TSV paper. As the TSV showed outstanding performance on detecting hallucinated answers from LLMs, we propose that the TSV can be used to guide hallucination mitigation. 

Besides, we modify the classification objective function to make make the two distributions (truthful and hallucinated) to be as far as possible so that it can handle edge cases better. 


## Requirements

```bash
conda env create -f tsv.yml
```
---

## LLM response generation

Generate responses for each question to construct an unlabeled QA dataset in the wild.

```bash
bash gen.sh
```

---

## GT generation

Generate [BLEURT](https://arxiv.org/abs/2004.04696) score for each QA pair


```bash
bash gt.sh
```

---

## Train TSV

Train TSV for hallucination detection.

```bash
bash train.sh
```

---

## Acknowledgement

We gratefully acknowledge [TSV](https://arxiv.org/abs/2503.01917), [ITI](https://arxiv.org/abs/2306.03341), and [ICV](https://arxiv.org/abs/2311.06668) for their inspiring ideas and open-source contributions, which served as valuable foundations for this work.
