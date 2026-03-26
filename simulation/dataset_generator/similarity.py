# similarity.py - simple histogram similarity
from PIL import Image
import numpy as np

def hist_similarity(imgA: Image.Image, imgB: Image.Image, bins=32):
    a = np.array(imgA.resize((128,128))).astype('float32')
    b = np.array(imgB.resize((128,128))).astype('float32')
    ha,_ = np.histogramdd(a.reshape(-1,3), bins=(bins,bins,bins), range=((0,255),(0,255),(0,255)))
    hb,_ = np.histogramdd(b.reshape(-1,3), bins=(bins,bins,bins), range=((0,255),(0,255),(0,255)))
    ha = ha.flatten(); hb = hb.flatten()
    ha = ha / (ha.sum()+1e-9); hb = hb / (hb.sum()+1e-9)
    inter = np.minimum(ha, hb).sum()
    return float(inter)
