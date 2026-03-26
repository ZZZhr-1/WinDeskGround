import re

def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return []
    return click_point

def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    if not matches:
        # Try finding raw (x,y),(x,y) pattern if box tags missing but implied
        pattern_raw = r"\((\d+,\d+)\),\((\d+,\d+)\)"
        matches = re.findall(pattern_raw, s)
        
    # Convert the tuples of strings into tuples of integers
    if matches:
        # Take the last match typically, or all? Reference impl takes sum?
        # Original: return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]
        # Simplify to return the first valid box found
        match = matches[-1] # Usually the last generation is the answer
        p1 = match[0].split(',')
        p2 = match[1].split(',')
        return [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))]
    return []
