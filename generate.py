#!/usr/bin/env python3
"""Generate and render Chinese trajectories. No training data or evaluator needed."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def validate(texts, vocabulary):
    errors=[]
    for n,text in enumerate(texts,1):
        if not text:
            errors.append({'line':n,'error':'empty text'})
        invalid=[{'position':i+1,'character':c,'unicode':f'U+{ord(c):04X}'}
                 for i,c in enumerate(text) if c not in vocabulary]
        if invalid: errors.append({'line':n,'text':text,'unknown':invalid})
    if errors:
        raise ValueError('字符检查失败；未替换或删除任何字符：\n'+json.dumps(errors,ensure_ascii=False,indent=2))


def verify_files():
    expected='14c59b130696de1eab207aecccb43ea334a00da6d7a871430c7028da318b4261'
    path=ROOT/'weights/ghdit_ep029.pth'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('权重校验失败：ghdit_ep029.pth')


def render(xy,path,width,height,linewidth,dpi,font=None,title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fig,ax=plt.subplots(figsize=(width,height))
    ax.plot(xy[:,0],xy[:,1],color='#244665',linewidth=linewidth,
            solid_capstyle='round',solid_joinstyle='round')
    ax.set_aspect('equal',adjustable='box')
    ax.margins(x=0.04,y=0.12)
    ax.axis('off')
    if title and font:
        ax.set_title(title,fontproperties=FontProperties(fname=font),fontsize=14)
    fig.tight_layout(pad=0.2)
    fig.savefig(path,dpi=dpi,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inp=p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--text',help='完整中文字符串，不会自动去除空格、标点')
    inp.add_argument('--text-file',type=Path,help='UTF-8，每行一句；不允许空行')
    inp.add_argument('--render-npz',type=Path,help='从已保存轨迹重绘，无需GPU')
    p.add_argument('--output-dir',type=Path,help='新输出目录；已有目录拒绝覆盖')
    p.add_argument('--check-only',action='store_true',help='只检查字符，不需要GPU或PyTorch')
    p.add_argument('--seed',type=int,default=20260920)
    p.add_argument('--samples-per-text',type=int,default=1)
    p.add_argument('--max-points',type=int,default=1200)
    p.add_argument('--max-characters',type=int,default=32)
    p.add_argument('--line-width',type=float,default=1.1)
    p.add_argument('--width',type=float,default=12)
    p.add_argument('--height',type=float,default=3)
    p.add_argument('--dpi',type=int,default=200)
    p.add_argument('--font',help='可选中文字体路径；提供时图片上方显示标签')
    a=p.parse_args()
    if min(a.samples_per_text,a.max_points,a.max_characters,a.dpi,a.line_width,a.width,a.height)<=0:
        p.error('数量、尺寸和线宽必须为正数')
    if a.font and not Path(a.font).is_file(): p.error('字体文件不存在')
    if not 0<=a.seed<2**63-1-a.samples_per_text: p.error('seed超出范围')
    if a.render_npz:
        if a.check_only: p.error('重绘不能与check-only一起使用')
        import numpy as np
        with np.load(a.render_npz,allow_pickle=False) as data:
            xy=data['xy'].copy()
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():
            p.error('轨迹应为有限值的[N,2]坐标')
        if not a.output_dir: p.error('请指定--output-dir')
        a.output_dir.mkdir(parents=True,exist_ok=False)
        render(xy,a.output_dir/'trajectory.png',a.width,a.height,a.line_width,a.dpi)
        return
    texts=[a.text] if a.text is not None else a.text_file.read_text(encoding='utf-8-sig').splitlines()
    if not texts: p.error('输入为空')
    alphabet=(ROOT/'assets/characters.txt').read_text(encoding='utf-8').splitlines()
    if len(alphabet)<3811 or len(set(alphabet))!=len(alphabet) or any(len(x)!=1 for x in alphabet):
        raise ValueError('字符表不符合原始类别索引')
    # Original dictionary contains extra suffix entries not covered by this checkpoint.
    # Preserve the original 1-based IDs; padding=0, valid embedding IDs=1..3811.
    vocabulary={c:i+1 for i,c in enumerate(alphabet[:3811])}
    validate(texts,vocabulary)
    if any(len(t)>a.max_characters for t in texts): p.error('字符串超过max-characters安全限制')
    if a.check_only:
        print(json.dumps({'valid':True,'labels':[[vocabulary[c] for c in t] for t in texts]},ensure_ascii=False))
        return
    if not a.output_dir: p.error('请指定--output-dir')
    if a.output_dir.exists(): p.error('输出目录已存在；请选择新目录以避免覆盖')
    import numpy as np
    import torch
    if not torch.cuda.is_available(): p.error('生成需要NVIDIA CUDA GPU；检查和重绘不需要GPU')
    # One visible/current GPU only; CUDA_VISIBLE_DEVICES selects a physical GPU.
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    verify_files()
    from ghdit.inference import Generator
    model=Generator(ROOT/'weights')
    lengths=[]
    for text in texts:
        labels=torch.tensor([[vocabulary[c] for c in text]],dtype=torch.long,device='cuda')
        raw=model.predict_length(labels)
        if not np.isfinite(raw) or not 7<=round(raw)<=a.max_points:
            raise ValueError(f'{text!r}预测长度{raw}超出[7,{a.max_points}]，未裁剪或生成')
        lengths.append((raw,round(raw)))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i,text in enumerate(texts):
        labels=torch.tensor([[vocabulary[c] for c in text]],dtype=torch.long,device='cuda')
        raw,points=lengths[i]
        for j in range(a.samples_per_text):
            seed=a.seed+i*a.samples_per_text+j
            offsets=model.generate(labels,points,seed)
            xy=np.cumsum(offsets,axis=0)
            safe=re.sub(r'[\\/:*?"<>|\s]','_',text)[:32]
            name=f'{i+1:04d}_{j+1:02d}__{safe}__seed-{seed}'
            np.savez_compressed(a.output_dir/(name+'.npz'),offsets=offsets,xy=xy,
                                label_ids=labels.cpu().numpy()[0])
            render(xy,a.output_dir/(name+'.png'),a.width,a.height,a.line_width,a.dpi,a.font,text)
            row={'text':text,'label_ids':labels.cpu().tolist()[0], 'seed':seed,
                 'predicted_length_raw':raw,'generation_points':points,
                 'trajectory':name+'.npz','image':name+'.png','checkpoint':'ghdit_ep029.pth',
                 'render':{'aspect':'equal','y_axis':'up','line_width':a.line_width,'dpi':a.dpi}}
            (a.output_dir/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
            rows.append(row)
            (a.output_dir/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row,ensure_ascii=False),flush=True)
    (a.output_dir/'COMPLETE').write_text(str(len(rows))+'\n')


if __name__=='__main__':
    try:
        main()
    except (ValueError,FileNotFoundError,FileExistsError) as e:
        print(str(e),file=sys.stderr)
        sys.exit(2)
