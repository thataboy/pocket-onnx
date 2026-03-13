from dataclasses import dataclass

@dataclass
class Config:
    models_dir: str = "models"
    tokenizer_path: str = "models/tokenizer.model"
    precision: str = "int8"
    temperature: float = 0.7
    eos_threshold: float = -4.0
    noise_clamp: float = 3.0
    lsd_steps: int = 1
    first_chunk_frames: int = 1
    max_chunk_frames: int = 15
    eos_extra_frames: int = 10
    sample_rate: int = 24000