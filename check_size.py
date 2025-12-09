import torch
import os
import pandas as pd
import time
from ultralytics import RTDETR

def inspect_model(weight_path, device='0'):
    if not os.path.exists(weight_path):
        print(f"? Path not found: {weight_path}")
        return None

    try:
        # 1. Load the model
        # map_location ensures we don't overload GPU immediately
        model = RTDETR(weight_path)
        
        # 2. Auto-detect Native Image Size
        try:
            train_args = model.ckpt.get('train_args', {})
            trained_imgsz = train_args.get('imgsz', 640) 
            if isinstance(trained_imgsz, list):
                trained_imgsz = trained_imgsz[0]
        except AttributeError:
            trained_imgsz = 640 

        print(f"--> Loading {os.path.basename(weight_path)} (Native Size: {trained_imgsz})")

        # 3. MANUAL Parameter Counting (Pure PyTorch)
        # This bypasses model.info() return issues
        try:
            # Access the underlying PyTorch model
            pytorch_model = model.model
            n_params = sum(p.numel() for p in pytorch_model.parameters())
            params_m = n_params / 1e6
        except Exception as e:
            print(f"   Warning: Could not count params manually ({e})")
            params_m = 0

        # 4. GFLOPs (Attempt to get it, but default to N/A if fails)
        try:
            # We try the built-in info, but catch the NoneType error
            info = model.info(detailed=False, verbose=False)
            if info is not None:
                gflops = info[3]
            else:
                gflops = "N/A"
        except:
            gflops = "N/A"

        # 5. Measure Real-world Latency (The most important metric)
        if torch.cuda.is_available():
            model.to(device)
            dummy_input = torch.rand(1, 3, trained_imgsz, trained_imgsz).to(device)
            
            # Warmup
            for _ in range(10):
                _ = model(dummy_input, verbose=False)
            
            # Timing
            start_time = time.time()
            iters = 50 # 50 iterations for average
            for _ in range(iters):
                _ = model(dummy_input, verbose=False)
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_latency_ms = ((end_time - start_time) / iters) * 1000
        else:
            avg_latency_ms = 0.0

        # 6. Disk Size
        disk_size = os.path.getsize(weight_path) / (1024 * 1024)

        return {
            "Model Name": os.path.basename(weight_path),
            "Native Imgsz": trained_imgsz,
            "Params (M)": round(params_m, 2),
            "GFLOPs": gflops, 
            "Latency (ms)": round(avg_latency_ms, 2),
            "Disk Size (MB)": round(disk_size, 2)
        }

    except Exception as e:
        print(f"? Error processing {os.path.basename(weight_path)}: {e}")
        return None

if __name__ == "__main__":
    # --- INPUT YOUR MODEL PATHS HERE ---
    model_files = [
        "/project/lt200246-mmacma/nuke/swamp/UAV-DETR/train1280_pad/exp6/weights/best.pt"
        # Add more paths here...
    ]
    
    print(f"Inspecting {len(model_files)} models on GPU...")
    
    results = []
    for path in model_files:
        res = inspect_model(path)
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(df.to_string(index=False))
        print("="*80)