# compositor.py - compose wallpaper + windows (simple alpha composite)
from PIL import Image, ImageOps, ImageFilter

def composite_desktop(wallpaper, windows):
    canvas = wallpaper.convert('RGBA').copy()
    layer = Image.new('RGBA', canvas.size, (0,0,0,0))
    for w in windows:
        img = w['window_image']
        x,y,wid,hei = w['placed_bbox']
        win_r = img.resize((wid,hei))
        layer.alpha_composite(win_r, (x,y))
    out = Image.alpha_composite(canvas, layer)
    return out.convert('RGB')
