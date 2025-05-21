import av
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import types
np.random.seed(0)
from torchvision import transforms

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria




def main():
    np.random.seed(0)
    ###
    def read_video_pyav(path, num_frames=16):
        '''
        Decode the video with PyAV and return a list of PIL.Image.
        '''
        container = av.open(path)
        stream = container.streams.video[0]
        total_frames = stream.frames or 100  # fallback if frame count is unknown
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_image())  # Convert to PIL.Image
            if len(frames) >= num_frames:
                break
        return frames
    # 🟩 替換為你的本地影片路徑
    video_path = "/my_workspace/Data/videos/00001.mp4"
    frames = read_video_pyav(video_path)
    # 🟦 替換為你的新聞句子
    text = [
        "關心昨天晚間6點36分，花蓮富里鄉、發生芮氏規模5.8地震，幾乎全台有感，所幸都沒有傳出嚴重災情，氣象署地震測報中心不排除今明後三天，還可能會有規模4.5到5.5的餘震，大家多加留意＃。",
        "關心早上地牛翻身，07:05花蓮外海發生規模6.2的有感地震，震度3級以上地區，包括有宜蘭、花蓮、新北、新竹、嘉義、雲林、彰化。更多地震消息，請鎖定公視各節新聞。",
        "嘉義縣新港鄉昨天下午五點半，發生芮氏規模5.5地震，深度8.5公里，地震明顯，使得高鐵兩班列車暫時停車，嘉義民雄鄉有民宅圍牆被震倒，還有雲林古坑鄉149甲線內湖明隧道則發生坍方落石，所幸無人受傷。",
        "兔年春節10天連假已經結束，今天開始上班，不少人早上起不來，今年過年冷颼颼，寒流發威，昨天清晨苗栗頭屋出現4度低溫，是入冬以來最低氣溫紀錄。光是1月27日到1月28日二天，全台超過140人，疑似因天冷猝死。氣象局指示，今天寒流雖然開始減弱，不過，明後天，各地都還是日夜溫差大，大家出門要做好保暖工作。",
        "關心地震消息", 
        "今天晚餐吃什麼", 
        "歡迎收看公視新聞"]
    # 🟨 載入多模態 XCLIP 模型（large 版本支援 vision-text）
    processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14-16-frames")
    XCLIP_model = AutoModel.from_pretrained("microsoft/xclip-large-patch14-16-frames").to("cuda").eval()
    
    ### 定選裡面的東西
    def hook_visual_proj_input(module, input, output):
        print(f"[🎯 visual_projection input] shape = {input[0].shape}")
        module._cached_input = input[0].detach().cpu()

    handle = XCLIP_model.visual_projection.register_forward_hook(hook_visual_proj_input)
    ###
    video_inputs = processor.image_processor(
        images=[frames],
        return_tensors="pt",
        size={"height": 336, "width": 336}, 
        do_resize=True,
        do_center_crop=False  
    )
    text_inputs = processor.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    '''
    print(video_inputs["pixel_values"].shape)
    print("---")
    print(text_inputs["input_ids"].shape)
    '''
    inputs = {
        "pixel_values": video_inputs["pixel_values"].to("cuda"),
        "input_ids": text_inputs["input_ids"].to("cuda"),
        "attention_mask": text_inputs["attention_mask"].to("cuda")
    }
    # 前向推論
    with torch.no_grad():
        outputs = XCLIP_model(**inputs)

    # 這行替代原本的 outputs.video_embeds
    video_tensor = XCLIP_model.visual_projection._cached_input.to(dtype=torch.float16).to(XCLIP_model.device)  # [1, 16, 768]


    print("XCLIP結束")

    print("LLaVA開始")
    disable_torch_init()
    inp = 'This is a sign language video. Please translate it into English.'
    '''
    video = 'videollava/serve/examples/sample_demo_1.mp4'
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    '''
    # ✅ 改為本地模型路徑
    model_path = '/my_workspace/Video-LLaVA-7B'
    cache_dir = 'cache_dir'  # ✅ 本地模型不用設定 cache_dir
    device = 'cuda'
    load_4bit, load_8bit = True, False  # 建議本地先用 fp16，穩定性較高
    model_name = 'Video-LLaVA-7B'
    print("LLaVA載入模型") 
    tokenizer, LLaVA_model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    print("LLaVA結束載入")
    #
    #video_processor = processor['video']
    #
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    #
    #print(LLaVA_model)
    #print(type(LLaVA_model))
    #print(LLaVA_model.model)
    #print(type(LLaVA_model.model))
    
    # 通過 LLaVA 的投影層轉成 [1, 16, 4096]
    video_tensor = LLaVA_model.model.mm_projector(video_tensor).unsqueeze(0)

    print("tensor.shape =",video_tensor.shape)
    
    print(f"{roles[1]}: {inp}")
    num_tokens = video_tensor.shape[1]
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * num_tokens) + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    print("input_ids.shape + video_tensor.shape",input_ids.shape,video_tensor.shape)
    with torch.inference_mode():
        output_ids = LLaVA_model.generate(
            input_ids,
            inputs_embeds=video_tensor,  # ✅ 改這行
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    print("LLaVA outputs 產生")
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print("LLaVA outputs 結束")
    print("Final output",outputs)

if __name__ == '__main__':
    main()