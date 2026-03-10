import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from compressai.zoo import cheng2020_attn, cheng2020_anchor
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def setup_env():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/compressed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Environment ready. Using device: {device}")
    return device

def compress_and_save(input_dir, output_base, device):
    results = []
    models_configs = {
        'attention': cheng2020_attn,
        'anchor': cheng2020_anchor
    }
    
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for m_name, m_func in models_configs.items():
        for q in range(1, 7):
            print(f"\nLoading Model: {m_name} | Quality: {q}")
            model = m_func(quality=q, pretrained=True).eval().to(device)
            
            out_path = os.path.join(output_base, m_name, f"q{q}")
            os.makedirs(out_path, exist_ok=True)
            
            for img_name in tqdm(images, desc=f"Quality {q}"):
                img_path = os.path.join(input_dir, img_name)
                
                img = Image.open(img_path).convert('RGB')
                x = transforms.ToTensor()(img).unsqueeze(0).to(device)
                
                h, w = x.size(2), x.size(3)
                p = 64
                new_h = (h + p - 1) // p * p
                new_w = (w + p - 1) // p * p
                padding = (0, new_w - w, 0, new_h - h)
                x_padded = F.pad(x, padding, mode='constant', value=0)
                
                

                with torch.no_grad():
                    out_enc = model.compress(x_padded)
                    
                    num_pixels = h * w
                    bpp = sum(len(s[0]) for s in out_enc['strings']) * 8.0 / num_pixels
                    
                    out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
                    x_hat = out_dec['x_hat'][:, :, :h, :w]
                
                rec_img = transforms.ToPILImage()(x_hat.squeeze().cpu().clamp(0, 1))
                rec_img.save(os.path.join(out_path, img_name))
                
                results.append({'file': img_name, 'model': m_name, 'q': q, 'bpp': bpp})
                
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return results

if __name__ == "__main__":
    dev = setup_env()
    raw = "data/raw"
    comp = "data/compressed"
    
    if not os.listdir(raw):
        print(f"Dir {raw} is empty!")
    else:
        data = compress_and_save(raw, comp, dev)
        pd.DataFrame(data).to_csv("results/stats.csv", index=False)
        print("\nCheck results/stats.csv and data/compressed/")