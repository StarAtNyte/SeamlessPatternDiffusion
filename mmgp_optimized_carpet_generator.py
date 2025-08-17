import json
import base64
import io
import os
import psutil
from pathlib import Path
from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional, Dict, Any
import random
from typing import List
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# MMGP Memory Management
try:
    from mmgp import offload, profile_type
    MMGP_AVAILABLE = True
    print("MMGP available - memory optimization enabled")
except ImportError:
    MMGP_AVAILABLE = False
    print("MMGP not available - install with: pip install mmgp")

# Model configuration - using SD3.5 Large optimized for RTX 4090
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DTYPE = torch.bfloat16

# RTX 4090 optimizations
RTX_4090_VRAM_GB = 24
SYSTEM_RAM_GB = psutil.virtual_memory().total / (1024**3)

# Local paths
MODELS_DIR = Path("./models")
OUTPUT_DIR = Path("./outputs")

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Determine optimal MMGP profile based on system specs
def get_optimal_mmgp_profile():
    """Determine the best MMGP profile for RTX 4090 based on system RAM."""
    if not MMGP_AVAILABLE:
        return None
    
    print(f"System RAM: {SYSTEM_RAM_GB:.1f}GB, GPU: RTX 4090 (24GB VRAM)")
    
    if SYSTEM_RAM_GB >= 48:
        print("Using HighRAM_HighVRAM profile - fastest performance")
        return profile_type.HighRAM_HighVRAM
    elif SYSTEM_RAM_GB >= 32:
        print("Using LowRAM_HighVRAM profile - balanced for RTX 4090")
        return profile_type.LowRAM_HighVRAM
    else:
        print("Using VeryLowRAM_LowVRAM profile - safest option")
        return profile_type.VeryLowRAM_LowVRAM

# RTX 4090 specific optimizations
def optimize_for_rtx_4090():
    """Apply RTX 4090 specific optimizations."""
    if torch.cuda.is_available():
        # Enable memory efficiency for RTX 4090
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to leave room for MMGP operations
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("RTX 4090 optimizations applied: TF32, reduced precision, memory fraction")
    else:
        print("CUDA not available - skipping RTX 4090 optimizations")

# Pattern Diffusion seamless generation functions
def asymmetricConv2DConvForward_circular(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    self.paddingX = (
        self._reversed_padding_repeated_twice[0],
        self._reversed_padding_repeated_twice[1],
        0,
        0
    )

    self.paddingY = (
        0,
        0,
        self._reversed_padding_repeated_twice[2],
        self._reversed_padding_repeated_twice[3]
    )
    working = F.pad(input, self.paddingX, mode="circular")
    working = F.pad(working, self.paddingY, mode="circular")

    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


# Sets the padding mode to circular on Conv2d
def make_seamless(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Handle LoRA compatible layers if they exist
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = asymmetricConv2DConvForward_circular.__get__(module, Conv2d)


# Sets the padding mode back to default on Conv2d
def disable_seamless(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Handle LoRA compatible layers if they exist
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = nn.Conv2d._conv_forward.__get__(module, Conv2d)


class MMGPOptimizedCarpetGenerator:
    def __init__(self):
        self.model_id = MODEL_ID
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mmgp_profile = get_optimal_mmgp_profile()
        
        # Apply RTX 4090 optimizations
        optimize_for_rtx_4090()
        
        print(f"Using device: {self.device}")
        print(f"MMGP profile: {self.mmgp_profile}")

    def initialize_pipeline(self):
        """Initialize SD3.5 Large pipeline with MMGP optimizations and fine-tuned LoRA."""
        if self.pipe is None:
            from diffusers import StableDiffusion3Pipeline
            
            print("Loading SD3.5 Large model with MMGP optimizations...")
            
            # Load model to CPU first for MMGP optimization
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if DTYPE == torch.float16 and self.device == "cuda" else None
            ).to("cpu")  # Load to CPU first for MMGP
            
            # Load fine-tuned LoRA if available
            lora_config_path = MODELS_DIR / "adapter_config.json"
            lora_weights_path = MODELS_DIR / "adapter_model.safetensors"

            if lora_config_path.exists() and lora_weights_path.exists():
                print("Loading fine-tuned LoRA weights...")
                try:
                    self.pipe.load_lora_weights(str(MODELS_DIR))
                    self.pipe.fuse_lora()
                    print("Fine-tuned LoRA weights loaded and fused successfully")
                except Exception as e:
                    print(f"Error loading fine-tuned LoRA weights: {e}")
                    print("Continuing with base model...")
            else:
                print(f"No fine-tuned LoRA found at {lora_config_path}, using base model")

            # Apply MMGP memory optimization
            if MMGP_AVAILABLE and self.mmgp_profile:
                print(f"Applying MMGP optimization with profile: {self.mmgp_profile}")
                try:
                    offload.profile(self.pipe, self.mmgp_profile)
                    print("MMGP optimization applied successfully")
                except Exception as e:
                    print(f"Error applying MMGP optimization: {e}")
                    print("Falling back to standard GPU loading...")
                    self.pipe.to(self.device)
            else:
                # Standard loading without MMGP
                self.pipe.to(self.device)
                
            print("Model loaded and optimized successfully!")

    def get_memory_stats(self):
        """Get current memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
            stats["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        stats["system_ram_used"] = psutil.virtual_memory().used / 1024**3
        stats["system_ram_total"] = psutil.virtual_memory().total / 1024**3
        
        return stats

    def diffusion_callback(self, pipe, step_index, timestep, callback_kwargs):
        """Callback for seamless pattern generation using Pattern Diffusion techniques."""
        # Sets transformer and VAE to have circular padding on conv2d for last 20% of steps
        if step_index == int(pipe.num_timesteps * 0.8):
            make_seamless(pipe.transformer)
            make_seamless(pipe.vae)

        # Noise Rolling: For the first 80% of steps, this shifts the noise slightly and wraps around the edge
        # Keep shifts at (64, 64) for SD3.5 compatibility as requested
        if step_index < int(pipe.num_timesteps * 0.8):
            callback_kwargs["latents"] = torch.roll(callback_kwargs["latents"], shifts=(64, 64), dims=(2, 3))

        return callback_kwargs

    def generate_carpet_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = None,
        enable_seamless: bool = True
    ) -> tuple[Image.Image, int, dict]:
        """Generate a single carpet image from prompt with memory stats."""
        self.initialize_pipeline()
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Get memory stats before generation
        memory_stats_before = self.get_memory_stats()
        
        # Default negative prompt focused on quality
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, low resolution, distorted, warped, out of focus, "
                "soft focus, poor quality, pixelated, noisy, grainy, artifacts, "
                "jpeg artifacts, compression artifacts, muddy, unclear, fuzzy, "
                "photographic, realistic, photograph, 3D, dimensional, shadows, "
                "lighting effects, depth, perspective, people, faces, text, "
                "watermark, signature, frame, border"
            )
        
        # Enhance prompt for high-quality seamless carpet pattern generation
        enhanced_prompt = (
            f"{prompt}, seamless repeating carpet pattern design, high resolution, "
            "sharp details, crisp lines, high quality, ultra detailed, "
            "luxurious textile design, ornate decorative motifs, rich colors, "
            "traditional craftsmanship, intricate details, symmetrical design, "
            "tileable pattern, perfect for flooring, no visible seams, "
            "continuous ornamental pattern, decorative border elements, "
            "flat design illustration, vector art style, clean, precise"
        )
        
        print(f"Generating carpet image with seed {seed}")
        print(f"Prompt: {enhanced_prompt[:100]}...")
        print(f"Memory before generation - GPU: {memory_stats_before.get('gpu_memory_allocated', 0):.1f}GB")
        
        # Ensure seamless is disabled before starting (required for Pattern Diffusion technique)
        if enable_seamless:
            disable_seamless(self.pipe.transformer)
            disable_seamless(self.pipe.vae)
        
        # Clear cache before generation for optimal memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        dtype_for_autocast = DTYPE if self.device == "cuda" else torch.float32
        with torch.autocast(self.device, dtype=dtype_for_autocast):
            if enable_seamless:
                # Use Pattern Diffusion seamless generation with callback
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=self.diffusion_callback
                ).images[0]
            else:
                # Standard generation without seamless techniques
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
        
        # Get memory stats after generation
        memory_stats_after = self.get_memory_stats()
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result, seed, {
            "memory_before": memory_stats_before,
            "memory_after": memory_stats_after
        }


# Pydantic models for API
class GenerateCarpetRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for carpet pattern generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain features")
    width: int = Field(1024, ge=512, le=1536, description="Image width")
    height: int = Field(1024, ge=512, le=1536, description="Image height")
    num_inference_steps: int = Field(28, ge=10, le=50, description="Number of inference steps")
    guidance_scale: float = Field(3.5, ge=1.0, le=10.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    enable_seamless: bool = Field(True, description="Enable seamless pattern generation")


class GenerateCarpetResponse(BaseModel):
    image_base64: str = Field(..., description="Generated carpet image as base64 string")
    seed: int = Field(..., description="Seed used for generation")
    prompt: str = Field(..., description="Enhanced prompt used")
    generation_time: float = Field(..., description="Time taken to generate image in seconds")
    memory_stats: dict = Field(..., description="Memory usage statistics")


# Initialize FastAPI app
app = FastAPI(
    title="MMGP Optimized Carpet Pattern Generator API",
    description="Generate seamless carpet patterns using SD3.5 Large optimized for RTX 4090 with MMGP",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator_instance = None

@app.on_event("startup")
async def startup_event():
    global generator_instance
    print("Initializing MMGP optimized carpet generator for RTX 4090...")
    generator_instance = MMGPOptimizedCarpetGenerator()
    print("MMGP optimized carpet generator ready!")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the carpet pattern generator UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MMGP Optimized Carpet Pattern Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .optimization-badge {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
            border-radius: 20px;
            padding: 8px 16px;
            margin: 10px 5px;
            display: inline-block;
            font-size: 0.9em;
            color: #4CAF50;
        }
        
        .content {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 0;
            min-height: 600px;
        }
        
        .form-panel {
            background: #f8f9fa;
            padding: 30px;
            border-right: 1px solid #e9ecef;
        }
        
        .result-panel {
            padding: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #fff;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.2s;
        }
        
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {
            outline: none;
            border-color: #1a1a2e;
        }
        
        .form-group textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .generate-btn {
            width: 100%;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            border: none;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .tile-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .tile-btn:hover {
            background: #218838;
            transform: translateY(-1px);
        }
        
        .loading {
            text-align: center;
            color: #6c757d;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1a1a2e;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result-image {
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .result-info {
            margin-top: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }
        
        .memory-stats {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            font-size: 12px;
            border-left: 4px solid #1a1a2e;
        }
        
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #f5c6cb;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
            
            .form-panel {
                border-right: none;
                border-bottom: 1px solid #e9ecef;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MMGP Optimized Carpet Generator</h1>
            <p>RTX 4090 optimized with intelligent memory management</p>
            <div>
                <span class="optimization-badge">⚡ MMGP Enabled</span>
                <span class="optimization-badge">🎯 RTX 4090 Tuned</span>
                <span class="optimization-badge">🧠 Smart Memory</span>
            </div>
        </div>
        
        <div class="content">
            <div class="form-panel">
                <form id="carpetForm">
                    <div class="form-group">
                        <label for="prompt">Carpet Design Prompt:</label>
                        <textarea id="prompt" placeholder="Describe your carpet design (e.g., Persian ornate patterns, geometric Islamic motifs)" required>Ornate Persian carpet design with intricate floral motifs</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="negativePrompt">Negative Prompt (optional):</label>
                        <textarea id="negativePrompt" placeholder="What to avoid in the carpet design"></textarea>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="width">Width:</label>
                            <input type="number" id="width" value="1024" min="512" max="1536" step="64">
                        </div>
                        <div class="form-group">
                            <label for="height">Height:</label>
                            <input type="number" id="height" value="1024" min="512" max="1536" step="64">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="steps">Steps:</label>
                            <input type="number" id="steps" value="28" min="10" max="50">
                        </div>
                        <div class="form-group">
                            <label for="guidance">Guidance:</label>
                            <input type="number" id="guidance" value="3.5" min="1.0" max="10.0" step="0.1">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="seed">Seed (optional):</label>
                        <input type="number" id="seed" placeholder="Random if empty">
                    </div>
                    
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="enableSeamless" checked> Enable Seamless Generation (SD3.5 Pattern Diffusion)
                        </label>
                    </div>
                    
                    <button type="submit" class="generate-btn" id="generateBtn">
                        🚀 Generate Optimized Pattern
                    </button>
                </form>
            </div>
            
            <div class="result-panel" id="resultPanel">
                <div class="loading" style="display: none;" id="loading">
                    <div class="spinner"></div>
                    <p>Generating your carpet pattern...</p>
                    <p><small>Using MMGP optimized SD3.5 Large on RTX 4090</small></p>
                    <p><small>Intelligent memory management active</small></p>
                </div>
                
                <div id="result" style="display: none;">
                    <img id="resultImage" class="result-image" alt="Generated carpet pattern">
                    <div class="result-info" id="resultInfo"></div>
                    <button id="tileBtn" class="tile-btn" style="margin-top: 15px;">
                        🔲 View Tiled Preview
                    </button>
                </div>
                
                <div id="tileResult" style="display: none;">
                    <img id="tiledImage" class="result-image" alt="Tiled carpet pattern preview">
                    <div class="result-info">
                        <strong>2x2 Tiled Preview</strong> - Check for seamless edges
                    </div>
                    <button id="backBtn" class="tile-btn" style="margin-top: 15px;">
                        ← Back to Original
                    </button>
                </div>
                
                <div id="error" class="error" style="display: none;"></div>
                
                <div id="placeholder" style="text-align: center; color: #6c757d;">
                    <h3>⚡ Ready to create</h3>
                    <p>MMGP optimized for maximum performance on RTX 4090</p>
                    <p><small>Intelligent memory management • Pattern Diffusion • LoRA support</small></p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('carpetForm');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const error = document.getElementById('error');
        const placeholder = document.getElementById('placeholder');
        const generateBtn = document.getElementById('generateBtn');
        const resultImage = document.getElementById('resultImage');
        const resultInfo = document.getElementById('resultInfo');

        function showElement(element) {
            [loading, result, error, placeholder, document.getElementById('tileResult')].forEach(el => el.style.display = 'none');
            element.style.display = element === loading ? 'block' : element === result ? 'block' : element === error ? 'block' : 'block';
        }

        function showError(message) {
            error.textContent = message;
            showElement(error);
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            showElement(loading);
            generateBtn.disabled = true;
            generateBtn.textContent = '⚡ Optimizing & Generating...';

            const formData = {
                prompt: document.getElementById('prompt').value,
                negative_prompt: document.getElementById('negativePrompt').value || null,
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                num_inference_steps: parseInt(document.getElementById('steps').value),
                guidance_scale: parseFloat(document.getElementById('guidance').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
                enable_seamless: document.getElementById('enableSeamless').checked
            };

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP ${response.status}`);
                }

                const data = await response.json();
                
                resultImage.src = `data:image/png;base64,${data.image_base64}`;
                
                const memoryInfo = data.memory_stats ? 
                    `<div class="memory-stats">
                        <strong>Memory Usage:</strong><br>
                        GPU: ${data.memory_stats.memory_after?.gpu_memory_allocated?.toFixed(1) || 'N/A'}GB / ${data.memory_stats.memory_after?.gpu_memory_total?.toFixed(1) || 'N/A'}GB<br>
                        RAM: ${data.memory_stats.memory_after?.system_ram_used?.toFixed(1) || 'N/A'}GB / ${data.memory_stats.memory_after?.system_ram_total?.toFixed(1) || 'N/A'}GB
                    </div>` : '';
                
                resultInfo.innerHTML = `
                    <strong>Seed:</strong> ${data.seed}<br>
                    <strong>Generation Time:</strong> ${data.generation_time.toFixed(1)}s<br>
                    <strong>Dimensions:</strong> ${formData.width}×${formData.height}<br>
                    <strong>Model:</strong> SD3.5 Large (MMGP Optimized)<br>
                    <strong>Optimizations:</strong> RTX 4090 + TF32 + Smart Memory<br>
                    <strong>Pattern Shifts:</strong> (64, 64) - SD3.5 optimized
                    ${memoryInfo}
                `;
                
                showElement(result);

            } catch (err) {
                console.error('Generation error:', err);
                showError(`Failed to generate carpet pattern: ${err.message}`);
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = '🚀 Generate Optimized Pattern';
            }
        });

        // Tile preview functionality
        document.getElementById('tileBtn').addEventListener('click', () => {
            const originalImage = document.getElementById('resultImage');
            const tiledImage = document.getElementById('tiledImage');
            
            // Create a canvas to tile the image 2x2
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width * 2;
                canvas.height = img.height * 2;
                
                // Draw the image 4 times in a 2x2 grid
                ctx.drawImage(img, 0, 0);
                ctx.drawImage(img, img.width, 0);
                ctx.drawImage(img, 0, img.height);
                ctx.drawImage(img, img.width, img.height);
                
                tiledImage.src = canvas.toDataURL();
                document.getElementById('tileResult').style.display = 'block';
                document.getElementById('result').style.display = 'none';
            };
            img.src = originalImage.src;
        });

        // Back to original button
        document.getElementById('backBtn').addEventListener('click', () => {
            document.getElementById('result').style.display = 'block';
            document.getElementById('tileResult').style.display = 'none';
        });
    </script>
</body>
</html>
"""

@app.get("/health")
async def health():
    """Health check endpoint"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy", 
        "model": MODEL_ID, 
        "model_type": "SD3.5 Large + MMGP Optimized + Optional Fine-tuned LoRA",
        "device": device,
        "mmgp_available": MMGP_AVAILABLE,
        "rtx_4090_optimized": True,
        "system_ram_gb": round(SYSTEM_RAM_GB, 1)
    }

@app.post("/generate", response_model=GenerateCarpetResponse)
async def generate_carpet_pattern(request: GenerateCarpetRequest):
    """Generate a seamless carpet pattern from text prompt with MMGP optimization"""
    global generator_instance
    
    if generator_instance is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        start_time = time.time()
        
        # Generate the carpet pattern with memory tracking
        image, actual_seed, memory_stats = generator_instance.generate_carpet_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            enable_seamless=request.enable_seamless
        )
        
        generation_time = time.time() - start_time
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Enhanced prompt used
        enhanced_prompt = (
            f"{request.prompt}, seamless repeating carpet pattern design, luxurious textile design, "
            "high-quality rug pattern, ornate decorative motifs, rich colors, "
            "traditional craftsmanship, intricate details, symmetrical design, "
            "tileable pattern, perfect for flooring, no visible seams, "
            "continuous ornamental pattern, decorative border elements, "
            "flat design illustration, no shadows, no 3D effects"
        )
        
        return GenerateCarpetResponse(
            image_base64=image_base64,
            seed=actual_seed,
            prompt=enhanced_prompt,
            generation_time=generation_time,
            memory_stats=memory_stats
        )
        
    except Exception as e:
        print(f"Error generating carpet pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model and optimizations"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "model_id": MODEL_ID,
        "model_type": "SD3.5 Large with MMGP optimization + optional fine-tuned LoRA",
        "dtype": str(DTYPE),
        "device": device,
        "rtx_4090_optimizations": {
            "tf32_enabled": True,
            "memory_fraction": 0.95,
            "cudnn_benchmark": True
        },
        "mmgp_optimization": {
            "available": MMGP_AVAILABLE,
            "profile": str(get_optimal_mmgp_profile()) if MMGP_AVAILABLE else None,
            "description": "Memory management for efficient GPU usage"
        },
        "system_specs": {
            "ram_gb": round(SYSTEM_RAM_GB, 1),
            "vram_gb": RTX_4090_VRAM_GB
        },
        "supports_seamless": True,
        "pattern_shifts": "(64, 64) - SD3.5 optimized",
        "max_dimensions": {"width": 1536, "height": 1536},
        "recommended_steps": 28,
        "recommended_guidance": 3.5,
        "lora_enabled": "Check models/ directory",
        "local_paths": {
            "models_dir": str(MODELS_DIR),
            "output_dir": str(OUTPUT_DIR)
        }
    }

@app.get("/memory-stats")
async def get_memory_stats():
    """Get current memory usage statistics"""
    global generator_instance
    
    if generator_instance:
        return generator_instance.get_memory_stats()
    else:
        # Basic stats without model loaded
        stats = {"system_ram_used": psutil.virtual_memory().used / 1024**3,
                "system_ram_total": psutil.virtual_memory().total / 1024**3}
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return stats


if __name__ == "__main__":
    print("Running MMGP optimized FastAPI carpet generator for RTX 4090...")
    print(f"System RAM: {SYSTEM_RAM_GB:.1f}GB")
    print(f"MMGP Available: {MMGP_AVAILABLE}")
    if MMGP_AVAILABLE:
        print(f"Optimal MMGP profile: {get_optimal_mmgp_profile()}")
    uvicorn.run("mmgp_optimized_carpet_generator:app", host="0.0.0.0", port=8000, reload=True)