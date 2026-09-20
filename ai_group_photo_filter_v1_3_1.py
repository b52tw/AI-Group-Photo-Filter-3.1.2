# -*- coding: utf-8 -*-
import os, sys, csv, shutil, threading, queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, numpy as np
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()

APP="AI合照快速篩選器 v1.3.1｜指定人物搜尋版"
YUNET="face_detection_yunet_2023mar.onnx"
SFACE="face_recognition_sface_2021dec.onnx"
EXTS={".jpg",".jpeg",".png",".bmp",".webp",".heic",".heif",".tif",".tiff"}

def resource(name):
    return Path(getattr(sys,"_MEIPASS",Path(__file__).parent))/name

def read_img(p):
    with Image.open(p) as im:
        im=ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size)>3200:
            s=3200/max(im.size); im=im.resize((int(im.width*s),int(im.height*s)))
        return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)

def unique(folder,p):
    d=folder/p.name; i=2
    while d.exists():
        d=folder/f"{p.stem}_{i}{p.suffix}"; i+=1
    return d

class AI:
    def __init__(self,score=.45):
        self.det=cv2.FaceDetectorYN.create(str(resource(YUNET)),"",(320,320),score,.3,5000)
        self.rec=cv2.FaceRecognizerSF.create(str(resource(SFACE)),"")
    def faces(self,img):
        h,w=img.shape[:2]; self.det.setInputSize((w,h))
        _,f=self.det.detect(img)
        return [] if f is None else list(f)
    def feature(self,img,face):
        return self.rec.feature(self.rec.alignCrop(img,face))
    def cosine(self,a,b):
        return float(self.rec.match(a,b,cv2.FaceRecognizerSF_FR_COSINE))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP)
        # v1.3.1: smaller default height + resizable, bottom action bar always visible.
        self.geometry("1000x680")
        self.minsize(820,600)

        self.src=tk.StringVar(); self.dst=tk.StringVar()
        self.mode=tk.StringVar(value="依人數篩選")
        self.people=tk.StringVar(value="只找合照：3人以上")
        self.sens=tk.StringVar(value="團體照優先")
        self.strict=tk.StringVar(value="標準")
        self.refs=[]; self.q=queue.Queue(); self.stop=False; self.prog=tk.DoubleVar()

        # Fixed bottom action bar is packed FIRST so it cannot be pushed off-screen.
        bottom=ttk.Frame(self,padding=(14,8))
        bottom.pack(side="bottom",fill="x")
        self.go=ttk.Button(bottom,text="開始 AI 篩選",command=self.start)
        self.go.pack(side="left")
        ttk.Button(bottom,text="停止",command=lambda:setattr(self,"stop",True)).pack(side="left",padx=8)
        ttk.Button(bottom,text="開啟輸出資料夾",command=self.openout).pack(side="right")

        main=ttk.Frame(self)
        main.pack(side="top",fill="both",expand=True)

        ttk.Label(main,text="AI 合照快速篩選器 v1.3.1",
                  font=("Microsoft JhengHei UI",20,"bold")).pack(pady=(12,1))
        ttk.Label(main,text="YuNet 人臉偵測＋SFace 指定人物比對｜本機離線處理｜照片不上傳").pack(pady=(0,2))
        ttk.Label(main,text="作者／程式所有人：峻爸",font=("Microsoft JhengHei UI",9)).pack(pady=(0,6))

        top=ttk.Frame(main); top.pack(fill="x",padx=14)
        for r,(lab,var,cmd) in enumerate([("來源照片：",self.src,self.picksrc),("輸出位置：",self.dst,self.pickdst)]):
            ttk.Label(top,text=lab,width=11).grid(row=r,column=0,pady=3)
            ttk.Entry(top,textvariable=var).grid(row=r,column=1,sticky="ew",padx=6)
            ttk.Button(top,text="選擇資料夾",command=cmd).grid(row=r,column=2)
        top.columnconfigure(1,weight=1)

        box=ttk.LabelFrame(main,text="搜尋模式")
        box.pack(fill="x",padx=14,pady=(8,4))
        ttk.Radiobutton(box,text="依人數篩選",variable=self.mode,value="依人數篩選").grid(row=0,column=0,padx=8,pady=5,sticky="w")
        ttk.Radiobutton(box,text="尋找指定人物",variable=self.mode,value="尋找指定人物").grid(row=0,column=1,padx=8,pady=5,sticky="w")
        ttk.Label(box,text="人數模式：").grid(row=1,column=0,padx=8,pady=4,sticky="e")
        ttk.Combobox(box,textvariable=self.people,state="readonly",width=27,
            values=["完整分類","只找合照：3人以上","只找團體照：6人以上","只找大合照：10人以上"]).grid(row=1,column=1,pady=4,sticky="w")
        ttk.Label(box,text="YuNet 靈敏度：").grid(row=1,column=2,padx=(18,5))
        ttk.Combobox(box,textvariable=self.sens,state="readonly",width=15,
            values=["精準優先","平衡","團體照優先"]).grid(row=1,column=3,padx=(0,8))

        ref=ttk.LabelFrame(main,text="指定人物參考照片（建議 2～5 張清楚正面照）")
        ref.pack(fill="x",padx=14,pady=4)
        ttk.Button(ref,text="＋ 加入參考照片",command=self.addrefs).grid(row=0,column=0,padx=8,pady=6)
        ttk.Button(ref,text="清除參考照片",command=self.clearrefs).grid(row=0,column=1,padx=4)
        ttk.Label(ref,text="比對嚴格度：").grid(row=0,column=2,padx=(22,5))
        ttk.Combobox(ref,textvariable=self.strict,state="readonly",width=11,
                     values=["嚴格","標準","寬鬆"]).grid(row=0,column=3)
        self.reftext=ttk.Label(ref,text="尚未加入參考照片")
        self.reftext.grid(row=1,column=0,columnspan=4,padx=8,pady=(0,6),sticky="w")

        ttk.Label(main,text="人物搜尋輸出：01_高度符合、02_疑似符合_人工確認。相似度僅供照片整理，不作身分證明。").pack(anchor="w",padx=18,pady=(3,2))
        ttk.Progressbar(main,variable=self.prog,maximum=100).pack(fill="x",padx=16,pady=(5,2))
        self.status=ttk.Label(main,text="請選擇來源照片資料夾")
        self.status.pack(anchor="w",padx=16,pady=(0,3))

        # Flexible log area with scrollbar; absorbs resize instead of action buttons.
        logframe=ttk.Frame(main)
        logframe.pack(fill="both",expand=True,padx=16,pady=(2,4))
        self.log=tk.Text(logframe,height=8,font=("Microsoft JhengHei UI",9),wrap="word")
        scroll=ttk.Scrollbar(logframe,orient="vertical",command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left",fill="both",expand=True)
        scroll.pack(side="right",fill="y")
        self.after(100,self.poll)

    def picksrc(self):
        p=filedialog.askdirectory()
        if p:
            self.src.set(p)
            if not self.dst.get(): self.dst.set(str(Path(p).parent/"AI照片篩選結果_v1.3.1"))
    def pickdst(self):
        p=filedialog.askdirectory()
        if p:self.dst.set(p)
    def addrefs(self):
        fs=filedialog.askopenfilenames(filetypes=[("照片","*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp *.tif *.tiff")])
        for f in fs:
            if f not in self.refs:self.refs.append(f)
        self.reftext.config(text=f"已加入 {len(self.refs)} 張："+"、".join(Path(x).name for x in self.refs[:5]))
    def clearrefs(self):
        self.refs=[]; self.reftext.config(text="尚未加入參考照片")
    def logmsg(self,s):
        self.log.insert("end",s+"\n"); self.log.see("end")
    def start(self):
        if not Path(self.src.get()).is_dir():
            messagebox.showwarning("提示","請先選擇來源照片資料夾。"); return
        if not self.dst.get():
            messagebox.showwarning("提示","請選擇輸出位置。"); return
        if self.mode.get()=="尋找指定人物" and not self.refs:
            messagebox.showwarning("提示","指定人物搜尋需要先加入參考人物照片。"); return
        self.stop=False; self.go.config(state="disabled")
        threading.Thread(target=self.scan,daemon=True).start()
    def scan(self):
        try:
            src,dst=Path(self.src.get()),Path(self.dst.get()); dst.mkdir(parents=True,exist_ok=True)
            score={"精準優先":.75,"平衡":.60,"團體照優先":.45}[self.sens.get()]
            ai=AI(score)
            files=[p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
            rows=[]; selected=0
            if self.mode.get()=="尋找指定人物":
                highdir=dst/"01_高度符合"; maybedir=dst/"02_疑似符合_人工確認"
                highdir.mkdir(exist_ok=True); maybedir.mkdir(exist_ok=True)
                refs=[]
                for rp in self.refs:
                    im=read_img(rp); fs=ai.faces(im)
                    if fs:
                        face=max(fs,key=lambda f:f[2]*f[3])
                        refs.append(ai.feature(im,face))
                if not refs: raise RuntimeError("參考照片中沒有偵測到可用的人臉，請改用清楚正面的照片。")
                high,maybe={"嚴格":(.50,.42),"標準":(.45,.36),"寬鬆":(.40,.32)}[self.strict.get()]
                self.q.put(("log",f"參考人臉：{len(refs)} 個；嚴格度：{self.strict.get()}。"))
                for i,p in enumerate(files,1):
                    if self.stop: break
                    try:
                        im=read_img(p); faces=ai.faces(im); best=-1
                        for face in faces:
                            feat=ai.feature(im,face)
                            best=max(best,max(ai.cosine(feat,r) for r in refs))
                        out=""; result="未符合"
                        if best>=high:
                            x=unique(highdir,p); shutil.copy2(p,x); out=str(x); result="高度符合"; selected+=1
                        elif best>=maybe:
                            x=unique(maybedir,p); shutil.copy2(p,x); out=str(x); result="疑似符合"; selected+=1
                        rows.append([str(p),p.name,len(faces),f"{best:.4f}" if best>=0 else "",result,out])
                    except Exception as e: rows.append([str(p),p.name,"","","辨識失敗",str(e)])
                    if i%5==0 or i==len(files): self.q.put(("p",100*i/max(1,len(files)),f"人物搜尋：{i}/{len(files)}"))
                headers=["原始路徑","檔名","偵測人臉數","最高相似度","判定","輸出路徑"]
            else:
                mode=self.people.get()
                limit={"完整分類":0,"只找合照：3人以上":3,"只找團體照：6人以上":6,"只找大合照：10人以上":10}[mode]
                cats=[("00_未偵測到人臉",0,0),("01_單人照",1,1),("02_雙人照",2,2),("03_小組照_3-5人",3,5),("04_團體照_6-10人",6,10),("05_大合照_10人以上",11,99999)]
                if limit==0:
                    for n,_,__ in cats:(dst/n).mkdir(exist_ok=True)
                else:
                    outdir=dst/{3:"只找合照_3人以上",6:"只找團體照_6人以上",10:"只找大合照_10人以上"}[limit]
                    outdir.mkdir(exist_ok=True)
                for i,p in enumerate(files,1):
                    if self.stop: break
                    try:
                        im=read_img(p); n=len(ai.faces(im)); cat=next(a for a,lo,hi in cats if lo<=n<=hi); out=""
                        if limit==0 or n>=limit:
                            d=dst/cat if limit==0 else outdir
                            x=unique(d,p); shutil.copy2(p,x); out=str(x); selected+=1
                        rows.append([str(p),p.name,n,cat,"已輸出" if out else "未達門檻",out])
                    except Exception as e: rows.append([str(p),p.name,"","辨識失敗",str(e),""])
                    if i%5==0 or i==len(files): self.q.put(("p",100*i/max(1,len(files)),f"人數篩選：{i}/{len(files)}"))
                headers=["原始路徑","檔名","偵測人臉數","分類","結果","輸出路徑"]
            with open(dst/"AI篩選結果_v1.3.1.csv","w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(headers); w.writerows(rows)
            self.q.put(("done",f"完成！共掃描 {len(files)} 張照片，輸出 {selected} 張候選照片。"))
        except Exception as e:self.q.put(("err",str(e)))
    def poll(self):
        try:
            while True:
                x=self.q.get_nowait()
                if x[0]=="p": self.prog.set(x[1]); self.status.config(text=x[2])
                elif x[0]=="log": self.logmsg(x[1])
                elif x[0]=="done":
                    self.go.config(state="normal"); self.logmsg(x[1]); messagebox.showinfo("完成",x[1])
                elif x[0]=="err":
                    self.go.config(state="normal"); messagebox.showerror("錯誤",x[1])
        except queue.Empty: pass
        self.after(100,self.poll)
    def openout(self):
        p=Path(self.dst.get())
        if p.exists(): os.startfile(p)
        else: messagebox.showinfo("提示","輸出資料夾尚未建立。開始篩選後會自動建立。")

if __name__=="__main__": App().mainloop()
