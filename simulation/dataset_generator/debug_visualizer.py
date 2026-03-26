# debug_visualizer.py - draw overlays for debugging
from PIL import Image, ImageDraw, ImageFont

def draw_debug_info(canvas, windows, show_bbox=True, show_label=True):
    debug_img = canvas.copy().convert('RGBA')
    draw = ImageDraw.Draw(debug_img, 'RGBA')
    for w in windows:
        # Only draw debug info for the target window
        if not w.get('is_target', False):
            continue
            
        x,y,wid,hei = w['placed_bbox']
        # Removed target window outline to focus only on click target
        # draw.rectangle([x,y,x+wid,y+hei], outline=(255,0,0,200), width=3)
        
        if show_bbox and 'target_bbox_proj' in w:
            bx1,by1,bx2,by2 = w['target_bbox_proj']
            # Draw the click target bbox
            draw.rectangle([bx1,by1,bx2,by2], outline=(0,255,0,255), width=4)
            # Fill removed as requested
            # draw.rectangle([bx1,by1,bx2,by2], fill=(0,255,0,60))
            
        if show_label:
            lab = f"TARGET: {w['meta']['type']} vis={w.get('visible_ratio',0):.2f}"
            draw.text((x+4,y+4), lab, fill=(255,255,255,255))
            
    return debug_img.convert('RGB')
