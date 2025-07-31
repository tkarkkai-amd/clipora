import pandas as pd
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import open_clip  # or open_clip
import numpy as np

from train import init_model
from clipora.config import parse_yaml_to_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_results(image_path, texts, before, after):
    # Load an image (left panel)
    img = Image.open(image_path)

    # Normalize values
    norm = Normalize(vmin=0, vmax=1)

    # Find max indices
    max_before_idx = np.argmax(before)
    max_after_idx = np.argmax(after)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    # Image panel
    img_ax = fig.add_axes([0.05, 0.1, 0.3, 0.8])
    img_ax.imshow(img)
    img_ax.axis('off')

    # Table panel
    table_ax = fig.add_axes([0.4, 0.1, 0.55, 0.8])
    table_ax.axis('off')

    # Table headers
    headers = ["Text", "Before", "After"]
    col_widths = [0.4, 0.3, 0.3]
    x_positions = np.cumsum([0] + col_widths[:-1])

    for i, header in enumerate(headers):
        table_ax.text(x_positions[i], 1, header, ha='left', va='bottom', fontsize=12, weight='bold', transform=table_ax.transAxes)

    # Draw cells
    for i, (text, b, a) in enumerate(zip(texts, before, after)):
        y = 0.9 - i * 0.08
        table_ax.text(x_positions[0], y, text, ha='left', va='center', fontsize=10, transform=table_ax.transAxes)

        for j, val in enumerate([b, a], start=1):
            color = plt.cm.Blues(norm(val)) if j == 1 else plt.cm.Reds(norm(val))
            x = x_positions[j]
            width = col_widths[j]
            rect = patches.Rectangle((x, y - 0.03), width, 0.06, transform=table_ax.transAxes,
                                    color=color, ec='red' if (j == 1 and i == max_before_idx) or (j == 2 and i == max_after_idx) else 'black', lw=1.5)
            table_ax.add_patch(rect)
            table_ax.text(x + width / 2, y, f"{val:.2f}", ha='center', va='center', fontsize=10, transform=table_ax.transAxes)

    plt.savefig("output_image.png", bbox_inches='tight')
    plt.close()

def main(original_model, lora_model, preprocess, config, csv_path=None):
    # === Load CSV and Select Data ===
    csv_path = csv_path or config.eval_dataset
    df = pd.read_csv(csv_path)
    # Get 10 random texts from csv as classes
    classes = df[config.text_col].drop_duplicates().sample(10).tolist()
    correct_text = classes[0]
    img_path = df[df[config.text_col] == correct_text][config.image_col].iloc[0]
    # === Preprocess Inputs ===
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text_tokens = open_clip.tokenize(classes).to(device)

    with torch.no_grad():
        img_feat_before = original_model.encode_image(image)
        txt_feat_before = original_model.encode_text(text_tokens)
        probs_before = (img_feat_before @ txt_feat_before.T).softmax(dim=-1).squeeze().cpu().numpy()

        img_feat_after = lora_model.encode_image(image)
        txt_feat_after = lora_model.encode_text(text_tokens)
        probs_after = (img_feat_after @ txt_feat_after.T).softmax(dim=-1).squeeze().cpu().numpy()

    print("probs before:")
    print(probs_before)
    print("probs after:")
    print(probs_after)

    visualize_results(img_path, classes, probs_before, probs_after)

if __name__ == "__main__":
    lora_adapter_path = "/workload/vlm-lora-finetune/src/clipora/bridge_output/final/"
    config = parse_yaml_to_config("/workload/vlm-lora-finetune/src/clipora/bridge_output/final/train_config.yaml")
    lora_model, preprocess = init_model(config, lora_adapter_path=lora_adapter_path)
    # Load the original CLIP model (no LORA)
    original_model, _, _ = open_clip.create_model_and_transforms(
        model_name=config.model_name,
        pretrained=config.pretrained,
    )
    
    original_model = original_model.to(device)
    lora_model = lora_model.to(device)
    main(original_model, lora_model, preprocess, config)