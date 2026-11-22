"""MiniCPM embeddings and OmniParser-v2 region proposals; no image fallback."""
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image


class MiniCPMEncoder:
    def __init__(self,path,device):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.float32 if device=='cpu' else torch.float16).to(device).eval()
        self.cache={};self.lock=Lock()

    def __call__(self,texts):
        with self.lock:return self._encode(texts)

    def _encode(self,texts):
        missing=list(dict.fromkeys(t for t in texts if t not in self.cache))
        for start in range(0,len(missing),8):
            group=missing[start:start+8]
            batch=self.tokenizer(group,max_length=512,padding=True,truncation=True,return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                hidden=self.model(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1).float()
                vectors=(hidden*mask).sum(1)/mask.sum(1)
                vectors=self.torch.nn.functional.normalize(vectors,p=2,dim=1).cpu().tolist()
            self.cache.update(zip(group,vectors))
        return [self.cache[t] for t in texts]


class RegionParser:
    """Use v2 detector + EasyOCR proposals; function semantics come from the VLM.

    Full upstream Florence captions are not used: memory construction needs boxes,
    and the separately logged before/action/after VLM call supplies the function.
    """
    def __init__(self,weights,device):
        from ultralytics import YOLO
        import easyocr
        self.device=device;self.detector=YOLO(str(weights))
        self.ocr=easyocr.Reader(['en'],gpu=False if device=='cpu' else device,verbose=False)

    def boxes(self,path):
        image=Image.open(path).convert('RGB')
        prediction=self.detector.predict(image,conf=.05,iou=.1,imgsz=640,device=self.device,verbose=False)[0]
        boxes=prediction.boxes.xyxy.cpu().tolist()
        for polygon,text,confidence in self.ocr.readtext(np.asarray(image),text_threshold=.9):
            if confidence>=.1:
                points=np.asarray(polygon);boxes.append([*points.min(0).tolist(),*points.max(0).tolist()])
        return boxes

    def crop_click(self,before,x,y,target):
        boxes=self.boxes(before)
        # A proposal must contain the click. Never invent a fixed square crop.
        candidates=[b for b in boxes if b[0]<=x<=b[2] and b[1]<=y<=b[3] and b[2]>b[0] and b[3]>b[1]]
        if not candidates:return None,boxes
        box=min(candidates,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        Image.open(before).crop(tuple(round(v) for v in box)).save(target)
        return box,boxes
