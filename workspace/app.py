from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import os

def main(image_path: str, prompt="Describe this image."):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name =  "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 出力ディレクトリが存在しない場合は作成
    output_dir = "workdir/vlm_captioning/workspace/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # キャプションをファイルに書き出す
    output_path = os.path.join(output_dir, "caption.txt")
    with open(output_path, "w") as f:
        f.write(caption)
    print(f"Caption saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 app.py <image_path>")
    else:
        main(sys.argv[1])