Cliva-Med: Lightweight Medical Vision-Language Model

Cliva-Med is a lightweight medical vision-language model (VLM) designed for healthcare and biomedical applications. It features a vision encoder, a lightweight large language model backbone, and a cross-modality projector. Cliva-Med is trained through a two-stage process: aligning multimodal medical images with language model tokens, and instruction fine-tuning on medical-specific datasets. Cliva-Med achieves state-of-the-art or competitive results on tasks like closed-ended medical visual question answering (VQA) and image classification — while using only 30-50% of the parameters compared to larger models.

    📄 Paper: Cliva-Med: Lightweight Medical Vision-Language Model

📦 Environment Setup

Clone the repository and move into the project directory:

git clone https://github.com/jun1299/Cliva-Med.git
cd Cliva-Med

Create and activate the environment:

conda create -n clivamed python=3.10 -y
conda activate clivamed
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

Replace the default router if needed:

    Download the domain-specific router provided or trained by yourself.

    Update its path in clivamed/model/language_model/llava_stablelm_moe.py.

📊 Training Datasets

Use the LLaVA-Med and Cliva-specific datasets:

    Alignment stage: LLaVA-Med Alignment Dataset

    Instruction Tuning: LLaVA-Med Instruct Dataset

    Image Download:

wget https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl
python download_image.py

Update your paths in config files.
🚀 Launch Web Interface

Use DeepSpeed to serve the Gradio interface:

Phi2-based Model

deepspeed --include localhost:0 clivamed/serve/gradio_web_server.py --model-path "./ClivaMed-phi2"

StableLM-based Model

deepspeed --include localhost:0 clivamed/serve/gradio_web_server.py --model-path "./ClivaMed-stablelm-1.6b"

🔍 Command Line Inference

Phi2-based

deepspeed --include localhost:0 clivamed/serve/cli.py --model-path "./ClivaMed-phi2" --image-file "image.jpg"

StableLM-based

deepspeed --include localhost:0 clivamed/serve/cli.py --model-path "./ClivaMed-stablelm-1.6b" --image-file "image.jpg"

🎛️ Model Zoo

    Stage1: Alignment

    Stage2: Instruction-Tuning

    Stage3: Expert Fine-tuning (if applicable)

📈 Evaluation

Example multi-GPU inference and evaluation workflow:

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

# Combine results
cat ./test_cliva-chunk2_{0..1}.jsonl > ./radvqa.jsonl

# Run evaluation
python run_eval.py --gt ./3vqa/test_rad.json --pred ./radvqa.jsonl --output ./data_RAD/wrong_answers.json

📚 Citation

@misc{jiang2024clivamed,
  title={Cliva-Med: Lightweight Medical Vision-Language Model},
  author={Songtao Jiang and Tuo Zheng and Yan Zhang and Yeying Jin and Li Yuan and Zuozhu Liu},
  year={2024},
  note={Available at https://github.com/jun1299/Cliva-Med}
}

🙏 Acknowledgements

Built upon the foundations of:

    MoE-LLaVA

    LLaVA-Med
