# sampler.py - sampling windows according to difficulty and inserting distractors
import random
from loader import load_window_image
from config import SCREEN_W, SCREEN_H

def sample_windows(metadata, difficulty_conf, image_root=None, path_to_meta=None, target_index=None):
    # Handle n_windows being a single int (from experiment config) or a tuple (from L1-L5 config)
    n_conf = difficulty_conf['n_windows']
    if isinstance(n_conf, int):
        n = n_conf
    else:
        n_min, n_max = n_conf
        n = random.randint(n_min, n_max)
    
    # Select target
    if target_index is not None:
        target = metadata[target_index % len(metadata)]
    else:
        target = random.choice(metadata)
    
    selected = [target]
    
    # Determine similar distractor based on sim_level
    sim_level = difficulty_conf.get('sim_level', 0)
    # Inject specific similar distractor
    if sim_level > 0:
        # consistent with requirement: Level 1 -> Rank 5 (index 4), Level 5 -> Rank 1 (index 0)
        rank_idx = 5 - sim_level
        if rank_idx < 0: rank_idx = 0
        distractor = get_ranked_distractor(target, rank_idx, path_to_meta)
        if distractor:
            # Avoid duplicates
            if distractor not in selected and len(selected) < n:
                selected.append(distractor)
    
    # Fill the rest with random windows
    attempts = 0
    while len(selected) < n and attempts < 1000:
        candidate = random.choice(metadata)
        if candidate not in selected:
            selected.append(candidate)
        attempts += 1
        
    return selected

def get_ranked_distractor(target_win, rank_idx, path_to_meta):
    # Use similar_contents from metadata if available
    if path_to_meta and 'similar_contents' in target_win and target_win['similar_contents']:
        valid_candidates = []
        target_path = target_win.get('image_path')
        # similar_contents is expected to be sorted by similarity descending
        for item in target_win['similar_contents']:
            p = item.get('data2_image_path')
            # Check if valid and not the target itself (though usually they are distinct)
            if p and p != target_path:
                if p in path_to_meta:
                    # Create a copy to avoid polluting the global metadata cache
                    # Inject the data2_mapping info which contains the specific bbox of the similar element
                    cand = path_to_meta[p].copy()
                    if 'data2_mapping' in item:
                        cand['distractor_element'] = item['data2_mapping']
                    valid_candidates.append(cand)
        if valid_candidates:
            if rank_idx < len(valid_candidates):
                return valid_candidates[rank_idx]
            else:
                # Fallback: if requested rank is not available (e.g. want 5th but only have 3),
                # return the least similar one available (last one) to honor "easier" intent if possible,
                # or just the available one.
                return valid_candidates[-1]

    return None
