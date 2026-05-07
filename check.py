# verify_hubert_stats.py
import torch
import torchaudio
import numpy as np
import librosa
import glob
import tarfile
import io
import os

# 1. HuBERT 로드
bundle = torchaudio.pipelines.HUBERT_LARGE
hubert = bundle.get_model().cuda().eval()

# 2. 사전 계산된 stat 로드
stat = np.load('/scratch/x3397a09/Amphion/models/vc/vevo/config/hubert_large_l18_mean_std.npz')
ref_mean = stat['mean']  # (1024,)
ref_std = stat['std']    # (1024,)

print(f"Reference mean: range [{ref_mean.min():.2f}, {ref_mean.max():.2f}], "
      f"avg={ref_mean.mean():.2f}")
print(f"Reference std:  range [{ref_std.min():.2f}, {ref_std.max():.2f}], "
      f"avg={ref_std.mean():.2f}")

# 3. Emilia 샘플 몇 개 뽑아서 HuBERT feature 추출
emilia_tars = sorted(glob.glob("/scratch/x3397a09/emilia_data_root/Emilia/EN/*.tar"))[:1]
all_feats = []

with tarfile.open(emilia_tars[0], 'r') as tar:
    members = [m for m in tar.getmembers() if m.name.endswith('.mp3')][:50]  # 50개만
    
    for m in members:
        f = tar.extractfile(m)
        if f is None: continue
        audio_bytes = f.read()
        
        try:
            with open('/tmp/_temp.mp3', 'wb') as tmp:
                tmp.write(audio_bytes)
            wav, sr = librosa.load('/tmp/_temp.mp3', sr=16000)
            if len(wav) < 16000: continue  # 너무 짧으면 skip
            
            wav_t = torch.tensor(wav).unsqueeze(0).cuda()
            with torch.no_grad():
                feats, _ = hubert.extract_features(wav_t, num_layers=18)
                feat = feats[-1].squeeze(0).cpu().numpy()  # (T, 1024)
            all_feats.append(feat)
        except Exception as e:
            print(f"Skip: {e}")
            continue

if all_feats:
    all_feats = np.concatenate(all_feats, axis=0)  # (total_T, 1024)
    print(f"\nEmilia samples: {len(all_feats)} frames from {len(members)} files")
    
    emilia_mean = all_feats.mean(axis=0)
    emilia_std = all_feats.std(axis=0)
    
    print(f"Emilia mean: range [{emilia_mean.min():.2f}, {emilia_mean.max():.2f}], "
          f"avg={emilia_mean.mean():.2f}")
    print(f"Emilia std:  range [{emilia_std.min():.2f}, {emilia_std.max():.2f}], "
          f"avg={emilia_std.mean():.2f}")
    
    # 비교
    mean_diff = np.abs(ref_mean - emilia_mean)
    std_ratio = emilia_std / (ref_std + 1e-8)
    
    print(f"\n--- Comparison ---")
    print(f"|ref_mean - emilia_mean|: avg={mean_diff.mean():.2f}, max={mean_diff.max():.2f}")
    print(f"emilia_std / ref_std: avg={std_ratio.mean():.2f}, "
          f"min={std_ratio.min():.2f}, max={std_ratio.max():.2f}")
    print(f"\nIf the stats match, mean_diff should be small and std_ratio close to 1.0")