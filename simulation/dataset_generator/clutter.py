# clutter.py - local/global clutter metrics and generation
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import random

def edge_density(image):
    edges = image.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges.convert('L'))
    return float((arr>30).mean())

def add_clutter(desktop_image, density_level='low'):
    
    draw = ImageDraw.Draw(desktop_image)
    w, h = desktop_image.size
    
    num_icons = 0
    if density_level == 'low': num_icons = 0
    elif density_level == 'mild': num_icons = 5
    elif density_level == 'medium': num_icons = 15
    elif density_level == 'high': num_icons = 30
    elif density_level == 'extreme': num_icons = 50
    
    for _ in range(num_icons):
        icon_w, icon_h = 48, 48
        # Prefer left side or top right
        if random.random() < 0.7:
            # Left side grid
            cx = random.randint(20, 300)
            cy = random.randint(20, h - 100)
        else:
            # Top right
            cx = random.randint(w - 300, w - 20)
            cy = random.randint(20, h//2)
            
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)
        draw.rectangle([cx, cy, cx+icon_w, cy+icon_h], fill=color, outline=None)
        
        # Add some text label simulation
        draw.rectangle([cx, cy+icon_h+5, cx+icon_w, cy+icon_h+15], fill=(255,255,255,200))

    return desktop_image

def add_notification(desktop_image):
    # Simulate a toast notification in bottom right
    w, h = desktop_image.size
    nw, nh = 300, 100
    nx = w - nw - 20
    ny = h - nh - 20
    
    draw = ImageDraw.Draw(desktop_image)
    
    # Draw shadow/bg
    draw.rectangle([nx, ny, nx+nw, ny+nh], fill=(30,30,30,230), outline=(100,100,100))
    
    # Draw "icon"
    draw.rectangle([nx+10, ny+10, nx+50, ny+50], fill=(0,120,215))
    
    # Draw "text" lines
    draw.rectangle([nx+60, ny+20, nx+200, ny+30], fill=(200,200,200))
    draw.rectangle([nx+60, ny+40, nx+250, ny+50], fill=(150,150,150))
    draw.rectangle([nx+60, ny+60, nx+180, ny+70], fill=(150,150,150))
    
    return desktop_image

