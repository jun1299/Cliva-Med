# Cliva-Med: Lightweight Medical Vision-Language Model

Cliva-Med is a lightweight medical vision-language model (VLM) designed for healthcare and biomedical applications. It integrates a **vision encoder**, a **lightweight large language model backbone**, and a **cross-modality projector**.

Cliva-Med is trained in two stages:
- **Alignment** of multimodal medical images and language tokens
- **Instruction fine-tuning** on medical-specific datasets

It achieves **state-of-the-art or competitive results** on tasks like **medical visual question answering (VQA)** and **image classification**, while using only **30-50% of the parameters** of larger models.



---

## 🔧 Installation

```bash
git clone https://github.com/jun1299/Cliva-Med.git
cd Cliva-Med

# Create and activate conda environment
conda create -n clivamed python=3.10 -y
conda activate clivamed

# Install dependencies
pip install --upgrade pip
pip install -e .

```

---

## 📦 Datasets

**Dataset Structure:**
- Alignment Stage: **PMC-VQA custom Alignment Dataset**
- Instruction Tuning: **LLaVA-Med Instruct Dataset**
**PMC-VQA custom Alignment Dataset**
  **Download images:**
```bash
  https://huggingface.co/datasets/RadGenome/PMC-VQA
```

**LLaVA-Med Instruct Dataset**
**Download images:**
```bash
wget https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl
python download_image.py
```


## 📊 Evaluation

Example multi-GPU inference and evaluation workflow:

```bash
CHUNKS=2
GPUS=(0 1)

for IDX in {0..1}; do
    GPU_IDX=${GPUS[$IDX]}
    PORT=$((${GPUS[$IDX]} + 29500))
    deepspeed --include localhost:$GPU_IDX --master_port $PORT model_vqa_med.py \
        --model-path your_model_path \
        --question-file ./test_rad.json \
        --image-folder ./3vqa/images \
        --answers-file ./test_cliva-chunk${CHUNKS}_${IDX}.jsonl \
        --temperature 0 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode stablelm/phi2 &
done
```

**Combine results:**
```bash
cat ./test_cliva-chunk2_{0..1}.jsonl > ./radvqa.jsonl
```

**Run evaluation:**
```bash
python run_eval.py --gt ./3vqa/test_rad.json --pred ./radvqa.jsonl --output ./data_RAD/wrong_answers.json
```



## 🙏 Acknowledgements

Built upon the outstanding works of:
- [Bunny-VLM](https://github.com/BAAI-DCAI/Bunny)
- [LLaVA-Med](https://github.com/OpenGVLab/LLaVA-Med)

---

## 📜 License

This project is licensed under the **Apache 2.0 License**.
