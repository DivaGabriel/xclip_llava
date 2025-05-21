import os
import cv2
import pandas as pd

# === 設定區 ===
csv_path = "CASE-pts-daily-20241114-224-1_484_classification.sign_language.csv"
video_path = "pts-daily-20241114-224-1.mp4"
output_dir = "output_gloss_clips"
os.makedirs(output_dir, exist_ok=True)

# 讀取標註 CSV
df = pd.read_csv(csv_path)

# 濾除 gloss 是空值的行
df = df.dropna(subset=["gloss"])

# 開啟影片
cap = cv2.VideoCapture(video_path)
print(video_path)
print(cap)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
print(width,height)
# 切割影片
i = 0
for idx, row in df.iterrows():
    start_frame = int(row["start"])
    end_frame = int(row["end"])
    gloss = str(row["gloss"]).strip().replace("/", "_")

    output_name = f"{i}_{gloss}.mp4"
    output_path = os.path.join(output_dir, output_name)
    i = i+1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_id in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    print(f"✔ Saved clip: {output_name} (frames {start_frame}-{end_frame})")

cap.release()
print("✅ 完成所有影片切割！")



'''
# 模型搬到 GPU
XCLIP_model.vision_model = XCLIP_model.vision_model.to("cuda")
with torch.no_grad():
    vit16_out = XCLIP_model.vision_model(pixel_values=pixel_values)  # ✅ OK
    patch16_features = vit16_out.last_hidden_state  # [8, num_tokens, 768]
print(patch16_features.shape)

xclippoorout = vit16_out.pooler_output

print(xclippoorout.shape)
video_embed = xclippoorout.unsqueeze(0)  # ➜ [1, 1, 4096] 這樣才能餵 LLaVA
print(video_embed .shape)
LLaVA_model.model.mm_projector = LLaVA_model.model.mm_projector.to("cuda")
projected = LLaVA_model.model.mm_projector(xclippoorout)
print(projected.shape)
video_embed_projected = projected.unsqueeze(0)  
print(video_embed_projected.shape)
'''