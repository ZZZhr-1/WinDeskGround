# config.py - global configuration for generator
SCREEN_W = 2560
SCREEN_H = 1440
SCREEN_AREA = SCREEN_W * SCREEN_H

DEFAULT_VISIBLE_RATIO = 0.20

CATEGORY_POSITION_PRIORS = {
    "Developer_Tools": (0.25, 0.55, 0.15, 0.45),
    "Advanced_Tools": (0.25, 0.55, 0.15, 0.45),
    "Productivity": (0.30, 0.60, 0.20, 0.50),
    "Browsers": (0.45, 0.80, 0.10, 0.50),
    "Communication": (0.70, 0.95, 0.55, 0.85),
    "Utilities": (0.05, 0.20, 0.60, 0.90),
    "Media&Entertainment": (0.20, 0.70, 0.15, 0.65),
    "Gaming": (0.10, 0.80, 0.10, 0.80),
    "File&System_Utilities": (0.05, 0.20, 0.60, 0.90),
}

WINDOW_SIZES = [
    (2560, 1440),
    (1280, 1440),
    (1280, 720),
    "original"
]

DIFFICULTY_LEVELS = {
    "L1": {
        "n_windows": (2, 4),
        "visible_min": 1.0,
        "visible_max": 1.0,
        "sim_level": 1
    },
    "L2": {
        "n_windows": (4, 6),
        "visible_min": 0.80,
        "visible_max": 0.90,
        "sim_level": 2
    },
    "L3": {
        "n_windows": (6, 9),
        "visible_min": 0.70,
        "visible_max": 0.80,
        "sim_level": 3
    },
    "L4": {
        "n_windows": (8, 12),
        "visible_min": 0.50,
        "visible_max": 0.70,
        "sim_level": 4
    },
    "L5": {
        "n_windows": (10, 15),
        "visible_min": 0.30,
        "visible_max": 0.50,
        "sim_level": 5
    },
}
