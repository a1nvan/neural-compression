import torch
import torchvision.transforms as transforms
from compressai.zoo import cheng2020_attention, cheng2020_anchor
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def run_compression_task(input_dir, output_base, model_type='attention'):
    device = 'cpu'
    results = []
    
    model_func = cheng2020_attention if model_type == 'attention' else cheng2020_anchor
    
    for q in range(1, 7):
        print(f"\n🚀 Processing {model_type} | Quality: {q}")
        model = model_func(quality=q, pretrained=True).eval().to(device)
        
        output_dir = os.path.join(output_base, model_type, f"q{q}")
        os.makedirs(output_dir, exist_ok=True)
        
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images):
            img_path = os.path.join(input_dir, img_name)
            save_path = os.path.join(output_dir, img_name)
            
           
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            
            with torch.no_grad():
                
                out_enc = model.compress(x)
                
        
                num_pixels = x.shape[2] * x.shape[3]
                bpp = sum(len(s[0]) for s in out_enc['strings']) * 8.0 / num_pixels
 
                out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
            
            rec_img = transforms.ToPILImage()(out_dec['x_hat'].squeeze().cpu().clamp(0, 1))
            rec_img.save(save_path)
            
            results.append({
                'filename': img_name,
                'model': model_type,
                'quality': q,
                'bpp': bpp
            })
            
    return results

if __name__ == "__main__":
    raw_path = "data/raw"
    comp_path = "data/compressed"
    os.makedirs("results", exist_ok=True)

    if not os.listdir(raw_path):
        print(f"Папка {raw_path} пуста! Положи туда хотя бы одну картинку.")
    else:
        all_data = []
        for m in ['anchor', 'attention']:
            all_data.extend(run_compression_task(raw_path, comp_path, model_type=m))
        
        # Сохраняем статистику для графиков
        df = pd.DataFrame(all_data)
        df.to_csv("results/compression_stats.csv", index=False)
        print("\nВсе готово! Метрики сохранены в results/compression_stats.csv")