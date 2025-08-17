# Pattern Diffusion - Seamless Carpet Pattern Generator

An advanced AI-powered carpet pattern generator using Stable Diffusion 3.5 Large with specialized seamless pattern generation techniques. This project implements the "Pattern Diffusion" method for creating tileable, seamless carpet designs with various optimization options.

## 🚀 Features

- **🎯 Seamless Pattern Generation**: Advanced "Pattern Diffusion" technique for perfectly tileable carpets
- **⚡ MMGP Optimization**: Memory-efficient generation optimized for RTX 4090 and consumer GPUs
- **🏛️ Fine-tuned Models**: Support for custom LoRA fine-tuning for specific carpet styles
- **🌐 Web Interface**: Beautiful FastAPI web interface for easy pattern generation
- **📊 Memory Monitoring**: Real-time GPU and RAM usage tracking
- **🔄 Batch Processing**: Generate multiple variations with different styles and parameters

## 📋 Available Versions

1. **`modal_fastapi_finetuned_carpet_generator.py`** - Original Modal cloud deployment
2. **`local_fastapi_carpet_generator.py`** - Local deployment version
3. **`mmgp_optimized_carpet_generator.py`** - MMGP optimized for RTX 4090 (Recommended)

## 🧠 Pattern Diffusion Method

### The Seamless Generation Technique

Our seamless pattern generation uses a novel approach called "Pattern Diffusion" that ensures perfect tileability:

#### 1. **Noise Rolling** (First 80% of steps)
```python
# Shifts noise by (64, 64) pixels and wraps around edges
if step_index < int(pipe.num_timesteps * 0.8):
    callback_kwargs["latents"] = torch.roll(callback_kwargs["latents"], shifts=(64, 64), dims=(2, 3))
```

#### 2. **Circular Padding** (Last 20% of steps)
```python
# Apply circular padding to Conv2D layers for seamless edges
if step_index == int(pipe.num_timesteps * 0.8):
    make_seamless(pipe.transformer)
    make_seamless(pipe.vae)
```

#### 3. **Custom Convolution Forward Pass**
```python
def asymmetricConv2DConvForward_circular(self, input, weight, bias):
    # Applies circular padding in both X and Y dimensions
    working = F.pad(input, self.paddingX, mode="circular")
    working = F.pad(working, self.paddingY, mode="circular")
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
```

This method ensures that:
- ✅ Patterns tile seamlessly without visible seams
- ✅ Maintains high visual quality and detail
- ✅ Works with SD3.5 Large's advanced transformer architecture
- ✅ Compatible with LoRA fine-tuning

## 🎨 Seamless Results Demonstration

### Example: Ornate Persian Carpet Pattern

<div align="center">

| Original Pattern | Tiled 2x2 Preview |
|:----------------:|:-----------------:|
| ![Original Pattern](generated_images/1.png) | ➡️ ![Tiled Pattern](generated_images/1_tiled.png) |

*Notice how the pattern tiles perfectly with no visible seams at the edges*

| Pattern 2 | Tiled 2x2 |
|:---------:|:---------:|
| ![Pattern 2](generated_images/2.png) | ➡️ ![Tiled 2](generated_images/2_tiled.png) |

| Pattern 3 | Tiled 2x2 |
|:---------:|:---------:|
| ![Pattern 3](generated_images/3.png) | ➡️ ![Tiled 3](generated_images/3_tiled.png) |

</div>

The arrows (➡️) show the transformation from single pattern to seamlessly tiled preview, demonstrating perfect edge continuity.

## 🛠️ Installation & Setup

### Requirements

```bash
# Install dependencies
pip install -r requirements_mmgp.txt

# Or individual packages:
pip install torch torchvision diffusers transformers accelerate peft safetensors Pillow fastapi uvicorn mmgp psutil xformers
```

### System Requirements

**Minimum:**
- GPU: RTX 3060 (12GB VRAM) or equivalent
- RAM: 16GB system RAM
- Storage: 20GB free space

**Recommended (MMGP Optimized):**
- GPU: RTX 4090 (24GB VRAM)
- RAM: 32-48GB system RAM
- Storage: 50GB free space

## 🚀 Quick Start

### 1. Run the MMGP Optimized Version (Recommended)

```bash
python mmgp_optimized_carpet_generator.py
```

### 2. Access the Web Interface

Open your browser to: `http://localhost:8000`

### 3. Generate Your First Pattern

1. Enter a carpet design prompt (e.g., "Ornate Persian carpet with intricate floral motifs")
2. Adjust dimensions and parameters
3. Enable "Seamless Generation" for tileable patterns
4. Click "Generate Optimized Pattern"

## 📁 Project Structure

```
Pattern-Diffusion/
├── mmgp_optimized_carpet_generator.py    # Main optimized version
├── local_fastapi_carpet_generator.py     # Local version
├── modal_fastapi_finetuned_carpet_generator.py  # Modal cloud version
├── requirements_mmgp.txt                 # Dependencies
├── models/                               # LoRA fine-tuned models
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── generated_images/                     # Example outputs
│   ├── 1.png
│   ├── 1_tiled.png
│   ├── 2.png
│   ├── 2_tiled.png
│   ├── 3.png
│   └── 3_tiled.png
├── generated-carpets-*/                  # Batch generation outputs
└── batch_generation/                     # Batch processing scripts
```

## ⚙️ Configuration Options

### MMGP Memory Profiles

The system automatically selects the optimal MMGP profile based on your hardware:

- **HighRAM_HighVRAM** (48GB+ RAM, 24GB+ VRAM) - Fastest
- **LowRAM_HighVRAM** (32GB+ RAM, 24GB+ VRAM) - Balanced for RTX 4090
- **VeryLowRAM_LowVRAM** (24GB+ RAM, 10GB+ VRAM) - Safest

### RTX 4090 Optimizations

Automatic optimizations applied:
- ✅ TF32 enabled for ~1.5x speed boost
- ✅ Reduced precision operations
- ✅ CuDNN benchmark mode
- ✅ 95% memory fraction allocation
- ✅ Smart memory cleanup

## 🎨 Generation Parameters

### Recommended Settings

| Parameter | Recommended | Range | Description |
|-----------|-------------|-------|-------------|
| **Steps** | 28 | 10-50 | Higher = better quality, slower |
| **Guidance** | 3.5 | 1.0-10.0 | Higher = more prompt adherence |
| **Dimensions** | 1024x1024 | 512-1536 | Must be multiples of 64 |
| **Seamless** | ✅ Enabled | - | Enable for tileable patterns |

### Prompt Engineering

**Good prompts:**
- "Ornate Persian carpet with intricate floral motifs"
- "Geometric Islamic patterns in deep blue and gold"
- "Traditional Turkish kilim with abstract designs"
- "Luxurious Victorian carpet with baroque elements"

**Avoid:**
- Photographic terms ("photo", "realistic")
- Lighting references ("shadows", "3D lighting")
- People or faces
- Text or watermarks

## 🔧 Advanced Usage

### Fine-tuning with LoRA

1. Place your fine-tuned LoRA files in the `models/` directory:
   - `adapter_config.json`
   - `adapter_model.safetensors`

2. The system will automatically detect and load them

### Batch Generation

Use the scripts in `batch_generation/` for large-scale pattern creation:

```bash
python batch_generation/modal_carpet_prompt_generator_sd35_1.py
```

### API Endpoints

- `GET /` - Web interface
- `POST /generate` - Generate pattern
- `GET /health` - System status
- `GET /model-info` - Model information
- `GET /memory-stats` - Memory usage

## 📊 Performance Benchmarks

### RTX 4090 Performance (MMGP Optimized)

| Resolution | Steps | Time (s) | VRAM Usage | Quality |
|------------|-------|----------|------------|---------|
| 1024x1024 | 28 | ~30-45 | ~18GB | Excellent |
| 1280x1280 | 28 | ~45-60 | ~22GB | Excellent |
| 1536x1536 | 28 | ~60-80 | ~23GB | Maximum |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the provided examples
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion 3.5 Large
- **MMGP** for memory optimization techniques
- **Pattern Diffusion** method for seamless generation
- **Hugging Face** for the Diffusers library

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce image dimensions
- Lower batch size
- Enable MMGP optimization
- Close other GPU applications

**Poor Seamless Quality:**
- Ensure "Enable Seamless Generation" is checked
- Use recommended prompt styles
- Avoid 3D/lighting terms in prompts
- Try different seeds

**Slow Generation:**
- Use MMGP optimized version
- Enable RTX 4090 optimizations
- Reduce number of steps
- Use lower precision (fp16)

### Getting Help

1. Check the Issues section on GitHub
2. Review the troubleshooting section
3. Ensure your system meets requirements
4. Try the example prompts first

---

<div align="center">

**🎨 Create Beautiful, Seamless Carpet Patterns with AI 🎨**

Made with ❤️ by the Pattern Diffusion Team

</div>
