import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers.generation import GenerationConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from utils import extract_bbox, pred_2_point
import re
import math
import json
import os
import base64
from dotenv import load_dotenv

load_dotenv()

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 100 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

class BaseModel:
    def __init__(self, model_path, device="cuda:0"):
        self.model_path = model_path
        self.device = device

    def predict(self, image_path, instruction, **kwargs):
        raise NotImplementedError

class OSAtlasModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        print(f"Loading OS-Atlas from {model_path}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device, 
            attn_implementation="flash_attention_2"
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.min_pixels = 4*28*28
        self.max_pixels = 1280*28*28 # Default max

    def predict(self, image_path, instruction, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": f"In this UI screenshot, what is the position of the element corresponding to the command \"click {instruction}\" (with bbox)?"},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        output_text_clean = output_text.split('<|box_start|>')[-1].split('<|box_end|>')[0]
        
        pred_point = None
        if 'box' in output_text_clean or '<box>' in output_text:
            pred_bbox = extract_bbox(output_text)
            if pred_bbox:
                # scale 1000
                pred_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2000, (pred_bbox[0][1] + pred_bbox[1][1]) / 2000]
        
        if not pred_point:
             pts = pred_2_point(output_text_clean)
             if pts:
                 pred_point = [x / 1000 for x in pts]
                 
        return pred_point, output_text

class UGroundModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        print(f"Loading UGround from {model_path}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device, attn_implementation="flash_attention_2"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        except:
            pass
        self.min_pixels = 4*28*28
        self.max_pixels = 1280*28*28

    def predict(self, image_path, instruction, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    {"type": "text", "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
Description: {instruction}
Answer:"""},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        pred_point = None
        if 'box' in output_text:
            pred_bbox = extract_bbox(output_text)
            if pred_bbox:
                pred_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2000, (pred_bbox[0][1] + pred_bbox[1][1]) / 2000]
        
        if not pred_point:
             pts = pred_2_point(output_text)
             if pts:
                 pred_point = [x / 1000 for x in pts]
                 
        return pred_point, output_text

class SeeClickModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        # SeeClick relies on Qwen-VL-Chat's tokenizer and config
        self.qwen_chat_path = '/data/home/zhr/models/Qwen-VL-Chat'
        
        print(f"Loading SeeClick from {model_path}...")
        print(f"Using base Qwen-VL-Chat components from {self.qwen_chat_path}")
        
        # Load Tokenizer from Qwen-VL-Chat
        self.tokenizer = AutoTokenizer.from_pretrained(self.qwen_chat_path, trust_remote_code=True)
        
        # Load Model from SeeClick checkpoint
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, bf16=True).eval()
        
        # Load GenerationConfig from Qwen-VL-Chat
        self.model.generation_config = GenerationConfig.from_pretrained(self.qwen_chat_path, trust_remote_code=True)

    def predict(self, image_path, instruction, **kwargs):
        prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"click {instruction}\" (with point)?"
        query = self.tokenizer.from_list_format([{'image': image_path}, {'text': prompt}])
        
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            
        pred_point = None
        if 'box' in response:
            try:
                pred_bbox = extract_bbox(response)
                if pred_bbox:
                     # Box output is typically 0-1 normalized floats
                     pred_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
            except:
                pass
        
        if not pred_point:
            pts = pred_2_point(response)
            if pts:
                # SeeClick point predictions are often 0-1 normalized floats (e.g., 0.5, 0.5)
                # But sometimes might be 0-1000. We check range.
                if any(x > 1.0 for x in pts):
                    pred_point = [x / 1000 for x in pts]
                else:
                    pred_point = pts
        
        return pred_point, response


class InfiGUIModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        print(f"Loading InfiGUI from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
        self.max_pixels = 1280 * 28 * 28

    def _resize_image(self, width: int, height: int, max_pixels: int) -> tuple[int, int]:
        current_pixels = width * height
        if current_pixels <= max_pixels:
            target_width, target_height = width, height
        else:
            scale_factor = math.sqrt(max_pixels / current_pixels)
            target_width = round(width * scale_factor)
            target_height = round(height * scale_factor)
        
        final_height, final_width = smart_resize(target_height, target_width)
        return final_width, final_height

    def predict(self, image_path, instruction, **kwargs):
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, ""

        original_width, original_height = image.size
        new_width, new_height = self._resize_image(original_width, original_height, self.max_pixels)
        resized_image = image.resize((new_width, new_height))
        
        system_prompt = 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.'
        prompt = f'''The screen's resolution is {new_width}x{new_height}.
Locate the UI element(s) for "{instruction}", output the coordinates using JSON format: [{{"point_2d": [x, y]}}, ...]'''

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Note: apply_chat_template expects list of messages for conversation
        # Depending on transformer version it might handle single conversation list or list of lists
        # User snippet used [messages] implying batch of 1 conversation.
        
        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=text, 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse output
        if "</think>" in output_text:
            content = output_text.split("</think>")[-1]
        else:
            content = output_text
            
        content = content.replace("```json", "").replace("```", "").strip()
        
        pred_point = None
        try:
            output_json = json.loads(content)
            if output_json and isinstance(output_json, list) and len(output_json) > 0:
                point_data = output_json[0]
                if "point_2d" in point_data:
                    coords = point_data["point_2d"]
                    x, y = coords[0], coords[1]
                    pred_point = [x / new_width, y / new_height]
        except:
            pass
            
        return pred_point, output_text


class SeedModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        from openai import OpenAI
        print(f"Loading Seed Model: {model_path}...")
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY"),
        )
        self.model_name = model_path

    def predict(self, image_path, instruction, **kwargs):
        # Encode image
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, ""
            
        prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"click {instruction}\"? Please output the point coordinates as (x, y)."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                reasoning_effort="low",
            )
            output_text = response.choices[0].message.content
        except Exception as e:
            print(f"Seed API Error: {e}")
            return None, ""

        pred_point = None
        # Parsing logic
        if not pred_point:
             pts = pred_2_point(output_text)
             # Check if 0-1000 or 0-1
             if pts:
                if any(x > 1.0 for x in pts):
                    pred_point = [x / 1000 for x in pts]
                else:
                    pred_point = pts
                    
        return pred_point, output_text


class UITARSAPIModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        from openai import OpenAI
        print(f"Loading UI-TARS API Model: {model_path}...")
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY"),
        )
        self.model_name = model_path

    def predict(self, image_path, instruction, **kwargs):
        # UI-TARS prompt
        prompt = f"""You are a GUI agent. You are given a task and a screenshot. You need to perform the next action to complete the task.

## Output Format
Action: ...

## Action Space
click(point='<point>x y</point>')

## User Instruction
click {instruction}"""

        # Encode image
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, ""
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            output_text = response.choices[0].message.content
        except Exception as e:
            print(f"UI-TARS API Error: {e}")
            return None, ""

        # UI-TARS Parsing Logic
        pred_point = None
        
        # 0. New UI-TARS format
        special_point_match = re.search(r"(?:point|start_box)=['\"]<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>['\"]", output_text)
        if special_point_match:
             pred_x, pred_y = float(special_point_match.group(1)), float(special_point_match.group(2))
             pred_point = [pred_x / width, pred_y / height]

        if not pred_point:
            box_match = re.search(r"start_box='\s*\(\s*(\d+),\s*(\d+)\s*\)\s*'", output_text)
            if box_match:
                pred_x, pred_y = float(box_match.group(1)), float(box_match.group(2))
                pred_point = [pred_x / width, pred_y / height]
        
        if not pred_point:
            point_match = re.search(r"<point>\s*(-?[\d\.]+)\s+(-?[\d\.]+)\s*</point>", output_text)
            if point_match:
                try:
                    x, y = float(point_match.group(1)), float(point_match.group(2))
                    if x > 1.0 or y > 1.0: 
                        pred_point = [x / width, y / height]
                    else:
                        pred_point = [x, y]
                except ValueError:
                    pass

        if not pred_point and 'box' in output_text:
             try:
                pred_bbox = extract_bbox(output_text)
                if pred_bbox:
                     pred_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2000, (pred_bbox[0][1] + pred_bbox[1][1]) / 2000]
             except: pass
             
        if not pred_point:
             pts = pred_2_point(output_text)
             if pts:
                if any(x > 1.0 for x in pts):
                    pred_point = [x / 1000 for x in pts]
                else:
                    pred_point = pts
        
        return pred_point, output_text


class UITARSModel(BaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        print(f"Loading UI-TARS from {model_path}...")
        
        # Use AutoModel with trust_remote_code=True to support Qwen2.5-VL or custom configs
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.max_pixels = 1280*28*28
        # self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    def predict(self, image_path, instruction, **kwargs):
        # UI-TARS official prompt adapted for single-step grounding
        prompt = f"""You are a GUI agent. You are given a task and a screenshot. You need to perform the next action to complete the task.

## Output Format
Action: ...

## Action Space
click(point='<point>x y</point>')

## User Instruction
click {instruction}"""
        
        # Calculate resized dimensions to decode model output
        image = Image.open(image_path)
        width, height = image.size
        # Use defaults matching qwen_vl_utils or explicit values if needed
        # Qwen2VL/UITARS default min_pixels is usually 4*28*28
        new_height, new_width = smart_resize(height, width, factor=28, min_pixels=4*28*28, max_pixels=self.max_pixels)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "max_pixels": self.max_pixels},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Explicitly set pad_token_id to suppress warnings
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, pad_token_id=pad_token_id)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        # UI-TARS output parsing
        pred_point = None
        
        # 0. New UI-TARS format: point='<|box_start|>(x,y)<|box_end|>' OR start_box='<|box_start|>(x,y)<|box_end|>'
        # Example from user: Action: click(start_box='<|box_start|>(783,344)<|box_end|>')
        special_point_match = re.search(r"(?:point|start_box)=['\"]<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>['\"]", output_text)
        if special_point_match:
             pred_x, pred_y = float(special_point_match.group(1)), float(special_point_match.group(2))
             # Map resized absolute coordinates back to normalized coordinates (0-1)
             # NOTE: smart_resize logic is strictly adhered to for W, H
             pred_point = [pred_x / new_width, pred_y / new_height]

        # 0.5 Try start_box='(x,y)' format per user feedback (legacy or alternative)
        # Handle case without special tokens if they are stripped or missing
        if not pred_point:
            box_match = re.search(r"start_box='\s*\(\s*(\d+),\s*(\d+)\s*\)\s*'", output_text)
            if box_match:
                pred_x, pred_y = float(box_match.group(1)), float(box_match.group(2))
                pred_point = [pred_x / new_width, pred_y / new_height]
        
        # 1. Try parsing specific <point> tag from UI-TARS prompt instruction (Assuming normalized if < 1, else absolute)
        # But UI-TARS v1.5 usually outputs relative 0-1000 or absolute pixels? 
        # Actually based on above, it seems it outputs ABSOLUTE PIXELS on resized image.
        if not pred_point:
            # Expected format: Action: click(point='<point>336 672</point>')
            # Regex to capture x and y which can be int or float, separated by space
            point_match = re.search(r"<point>\s*(-?[\d\.]+)\s+(-?[\d\.]+)\s*</point>", output_text)
            if point_match:
                try:
                    x, y = float(point_match.group(1)), float(point_match.group(2))
                    # Logic update: If it looks like absolute pixels (e.g. > 1), divide by new_width/height.
                    # If it looks like normalized 0-1, keep it.
                    if x > 1.0 or y > 1.0: 
                        pred_point = [x / new_width, y / new_height]
                    else:
                        pred_point = [x, y]
                except ValueError:
                    pass

        # 2. Fallback to box parsing
        if not pred_point and 'box' in output_text:
             try:
                pred_bbox = extract_bbox(output_text)
                if pred_bbox:
                     # extract_bbox usually handles 0-1000 range (e.g. Qwen2-VL style)
                     # For UI-TARS, check if it's absolute pixel box?
                     # If the box numbers are big (>1000), treat as absolute.
                     # But usually extract_bbox normalizes 0-1000 -> 0-1000.
                     # Let's assume standard Qwen 0-1000 normalization for now unless clearly indicated otherwise.
                     # Wait, user example showed absolute pixels in start_box.
                     pass 
                     pred_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2000, (pred_bbox[0][1] + pred_bbox[1][1]) / 2000]
             except: pass

             
        # 3. Fallback to raw number parsing
        if not pred_point:
             # Try standard coordinate parsing
             pts = pred_2_point(output_text)
             # Check if 0-1000 or 0-1
             if pts:
                if any(x > 1.0 for x in pts):
                     # Assume 0-1000 for standard fallback
                    pred_point = [x / 1000 for x in pts]
                else:
                    pred_point = pts
        
        return pred_point, output_text

def get_model(model_type, model_path):
    if model_type.lower() == 'os-atlas':
        return OSAtlasModel(model_path)
    elif model_type.lower() == 'uground':
        return UGroundModel(model_path)
    elif model_type.lower() == 'seeclick':
        return SeeClickModel(model_path)
    elif model_type.lower() == 'ui-tars':
        return UITARSModel(model_path)
    elif model_type.lower() == 'infigui':
        return InfiGUIModel(model_path)
    elif model_type.lower() == 'seed':
        return SeedModel(model_path)
    elif model_type.lower() == 'ui-tars-api':
        return UITARSAPIModel(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
