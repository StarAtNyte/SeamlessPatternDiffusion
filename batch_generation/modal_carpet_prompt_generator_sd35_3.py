import modal
import json
import base64
import io
import os
from pathlib import Path
from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional
import random
from typing import List, Dict
import time

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0", "torchvision>=0.16.0", "diffusers>=0.30.0", "transformers",
        "accelerate", "peft", "safetensors", "Pillow", "fastapi",
        "uvicorn", "python-multipart", "numpy<2.0", "tqdm", "sentencepiece"
    ])
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "git"])
)

app = modal.App("carpet-prompt-generator-sd35", image=image)

# Model configuration for SD3.5 finetuned
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"  # Replace with your finetuned model
DTYPE = torch.bfloat16

# Volume for storing generated images
model_volume = modal.Volume.from_name("carpet-model-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("generated-carpet-designs-sd35", create_if_missing=True)

# SD3.5-adapted seamless generation functions
def asymmetricConv2DConvForward_circular(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """Circular padding for Conv2d layers - adapted for SD3.5 architecture"""
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


# Sets the padding mode to circular on Conv2d - FLUX compatible
def make_seamless_sd35(model):
    """Enable circular padding on all Conv2d layers in SD3.5 model"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Handle LoRA compatible layers if they exist (FLUX may use them)
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = asymmetricConv2DConvForward_circular.__get__(module, Conv2d)


# Sets the padding mode back to default on Conv2d - FLUX compatible
def disable_seamless_sd35(model):
    """Disable circular padding and restore default Conv2d behavior"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Handle LoRA compatible layers if they exist
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = nn.Conv2d._conv_forward.__get__(module, Conv2d)


class SD35CarpetGenerator:
    def __init__(self):
        self.model_id = MODEL_ID
        self.pipe = None

    def initialize_pipeline(self):
        """Initialize SD3.5 pipeline with fine-tuned LoRA."""
        if self.pipe is None:
            from diffusers import StableDiffusion3Pipeline
            
            print("Loading SD3.5 model...")
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
                variant="fp16" if DTYPE == torch.float16 else None
            )
            
            # Load fine-tuned LoRA if available
            lora_config_path = "/models/adapter_config.json"
            lora_weights_path = "/models/adapter_model.safetensors"

            if Path(lora_config_path).exists() and Path(lora_weights_path).exists():
                print("Loading fine-tuned LoRA weights...")
                try:
                    self.pipe.load_lora_weights("/models/")
                    self.pipe.fuse_lora()
                    print("Fine-tuned LoRA weights loaded and fused successfully")
                except Exception as e:
                    print(f"Error loading fine-tuned LoRA weights: {e}")
                    print("Continuing with base model...")
            else:
                print(f"No fine-tuned LoRA found at {lora_config_path}, using base model")

            # Move to CUDA without quality-reducing optimizations
            self.pipe.to("cuda")
            print("SD3.5 model loaded successfully!")

    def sd35_diffusion_callback(self, pipe, step_index, timestep, callback_kwargs):
        """
        Callback for seamless pattern generation adapted for SD3.5 architecture.
        SD3.5 uses diffusion process, so we adapt the timing and approach.
        """
        # For SD3.5, we apply seamless techniques in the last 30% of steps
        # SD3.5 typically uses ~50 steps
        total_steps = pipe.num_inference_steps if hasattr(pipe, 'num_inference_steps') else 50
        late_stage_threshold = int(total_steps * 0.7)
        
        # Apply circular padding to UNet and VAE in late stages
        if step_index >= late_stage_threshold:
            # SD3.5 has UNet and VAE
            if hasattr(pipe, 'unet') and pipe.unet is not None:
                make_seamless_sd35(pipe.unet)
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                make_seamless_sd35(pipe.vae)

        # Noise rolling for early stages (adapted for SD3.5's diffusion)
        if step_index < late_stage_threshold:
            # SD3.5 uses latents with standard tensor structure
            if "latents" in callback_kwargs:
                latents = callback_kwargs["latents"]
                # Check latent tensor dimensions before rolling
                if len(latents.shape) >= 4:  # Should be [batch, channels, height, width]
                    # Standard shifts for SD3.5
                    shift_amount = min(32, max(8, latents.shape[-1] // 16))
                    # Use -2, -1 for last two dimensions (height, width) - FIXED BUG
                    callback_kwargs["latents"] = torch.roll(
                        latents, 
                        shifts=(shift_amount, shift_amount), 
                        dims=(-2, -1)
                    )

        return callback_kwargs

    def generate_carpet_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: int = None,
        enable_seamless: bool = True
    ) -> Image.Image:
        """Generate a single carpet image from prompt."""
        self.initialize_pipeline()
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Enhanced negative prompt optimized for seamless patterns
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, warped, photographic, realistic, "
                "photograph, 3D, dimensional, shadows, lighting effects, depth, "
                "perspective, people, faces, text, watermark, signature, frame, "
                "border, poor quality, pixelated, noisy, grainy, artifacts, "
                "carpet texture, fabric texture, weave, fibers, physical material, "
                "photography, studio lighting, product shot, wildlife photography, "
                "nature documentary, safari, zoo, animals in wild, natural habitat, "
                "realistic animals, photorealistic wildlife, animal photography, "
                "nature scene, landscape, outdoor scene, real animals, living creatures, "
                "naturalistic, documentary style, animal portrait, wildlife scene, "
                "environmental background, forest scene, jungle scene, savanna, "
                "natural environment, realistic fur, realistic feathers, realistic scales, "
                "flowers, floral, petals, blooms, blossoms, roses, tulips, daisies, "
                "botanical, garden, meadow, field of flowers, flower heads, flower buds, "
                "flowering plants, flowering vines, floral patterns, floral motifs, "
                "flower arrangements, bouquets, corsages, garlands, wreaths, "
                "lily, orchid, sunflower, peony, carnation, iris, hibiscus, jasmine, "
                "cherry blossom, sakura, lotus flower, water lily, poppy, lavender, "
                "marigold, chrysanthemum, magnolia, azalea, camellia, gardenia, "
                "flower garden, botanical garden, greenhouse, nursery, floriculture, "
                "petal-like, flower-like, blossom-like, bloom-shaped, "
                "seams, visible edges, discontinuity, mismatched patterns, "
                "asymmetric, uneven, irregular spacing, broken pattern"
            )
        
        
        print(f"Generating {'seamless ' if enable_seamless else ''}image with seed {seed}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Ensure seamless is disabled before starting (SD3.5 adaptation)
        if enable_seamless:
            if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
                disable_seamless_sd35(self.pipe.unet)
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                disable_seamless_sd35(self.pipe.vae)
        
        with torch.autocast("cuda", dtype=DTYPE):
            if enable_seamless:
                # Use SD3.5 seamless generation with callback
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=self.sd35_diffusion_callback
                ).images[0]
            else:
                # Standard SD3.5 generation without seamless techniques
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
        
        return result, seed

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": output_volume, "/models": model_volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_carpet_dataset():
    """Generate carpet dataset from prompts JSON file."""
    
    # Define artistic styles to prepend to prompts
    artistic_styles = [
        "Watercolor: soft translucent washes with organic bleeds and spontaneous edges",
        "Oil Paint: rich colors with thick impasto and visible brushstrokes",
        "Block Print: hand-carved aesthetic with bold silhouettes and registration shifts",
        "Ink Drawing: bold black linework with varied weights and crosshatching",
        "Digital Vector: mathematically precise curves with perfect color fills",
        "Batik: wax-resist crackle effects with organic color penetration",
        "Screen Print: flat uniform colors with razor-sharp edges",
        "Charcoal: deep blacks with smudged gradients and soft transitions",
        "Embroidery: raised stitching with varied textures and decorative patterns",
        "Pointillism: pure color dots creating optical mixing effects"
    ]
    
    # Test prompts optimized for FLUX seamless pattern generation
    prompts_data = {
  "Minimalistic": [
    "Single horizontal line in charcoal gray, minimal linear element with Zen aesthetic and Japanese ma spatial concept",
    "Three circles in perfect alignment in neutral tone, geometric minimalism with Bauhaus principle and modernist reduction philosophy",
    "Vertical stripe in navy blue, simple linear pattern with Scandinavian design and functional aesthetic restraint",
    "Square grid in pale gray, minimal geometric structure with Swiss design grid system and typographic precision",
    "Two-tone diagonal split in black and white, stark minimalism with De Stijl movement and absolute contrast",
    "Dotted line pattern in subtle beige, understated repetition with minimalist principle and quiet visual rhythm",
    "Single curve in soft white, organic minimalism with modernist sculpture and essential form reduction",
    "Rectangle frame in light gray, structural minimalism with architectural principle and spatial boundary definition",
    "Parallel lines in muted blue, linear minimalism with modernist aesthetic and geometric purity",
    "Corner accent in warm gray, architectural minimalism with building detail and structural accent point",
    "Circular void in neutral tone, negative space minimalism with spatial concept and absence as presence",
    "Thin border in pale silver, edge minimalism with frame concept and boundary definition subtlety",
    "Triangle point in monochrome, angular minimalism with directional element and geometric essence",
    "Gradient fade in soft gray, tonal minimalism with atmospheric effect and color transition subtlety",
    "Cross intersection in minimal black, linear junction with Swiss design and functional symbol reduction",
    "Wave line in light blue, organic minimalism with natural form and essential movement capture",
    "Grid dot in neutral beige, structural minimalism with measurement system and organizational principle",
    "Diagonal accent in charcoal, angular minimalism with dynamic element and directional emphasis",
    "Circular outline in pale gold, geometric minimalism with perfect form and essential shape definition",
    "Vertical division in soft white, spatial minimalism with architectural proportion and golden ratio",
    "Minimal texture in stone gray, surface minimalism with material essence and tactile suggestion",
    "Linear progression in gradient gray, sequential minimalism with mathematical order and visual rhythm",
    "Corner radius in warm white, architectural minimalism with building detail and modernist refinement",
    "Asymmetric balance in neutral tone, compositional minimalism with visual weight and spatial harmony",
    "Narrow stripe in pale blue, linear minimalism with functional decoration",
    "Rectangle sequence in monochrome, modular minimalism with systematic order and geometric progression",
    "Subtle shadow in soft gray, atmospheric minimalism with light effect and dimensional suggestion",
    "Edge detail in minimal silver, architectural minimalism with building trim and structural precision",
    "Geometric intersection in pale gold, mathematical minimalism with pure form and essential geometry",
    "Minimal border in neutral beige, frame minimalism with boundary definition and spatial containment",
    "Clean line in charcoal gray, essential minimalism with drawing reduction and linear purity",
    "Simple curve in soft white, organic minimalism with natural form and movement essence",
    "Grid structure in light gray, organizational minimalism with system logic and rational order",
    "Minimal accent in warm silver, decorative minimalism with essential ornament and functional beauty",
    "Geometric void in neutral tone, negative minimalism with spatial concept and emptiness as design",
    "Linear rhythm in pale blue, sequential minimalism with musical analogy and visual tempo",
    "Structural element in minimal black, architectural minimalism with building component and functional design",
    "Tonal variation in soft gray, atmospheric minimalism with color subtlety and perceptual refinement",
    "Essential form in warm white, sculptural minimalism with pure shape and material honesty",
    "Minimal detail in light gold, decorative minimalism with essential ornament and luxury restraint",
    "Geometric progression in neutral sequence, mathematical minimalism with logical order and visual calculation",
    "Spatial division in pale silver, architectural minimalism with proportion system and golden section",
    "Clean geometry in monochrome, pure minimalism with essential form and absolute reduction",
    "Minimal texture in stone beige, material minimalism with surface quality and tactile suggestion",
    "Linear sequence in soft gray, rhythmic minimalism with visual music and geometric tempo",
    "Essential detail in warm charcoal, architectural minimalism with building accent and structural poetry",
    "Geometric purity in neutral white, absolute minimalism with perfect form and essential geometry",
    "Minimal rhythm in pale blue sequence, musical minimalism with visual tempo and systematic beauty",
    "Structural poetry in light gray, architectural minimalism with building essence and spatial meditation",
    "Essential beauty in minimal gold, luxury minimalism with precious restraint and quiet elegance"
  ],
  "Organic": [
    "Stylized water ripple pattern in aquatic blue, decorative wave motif for carpet design with flowing curved lines, flat textile pattern",
    "Geometric river stone pattern in earth tones, decorative oval shapes arranged in organic clusters for rug design, two-dimensional",
    "Stylized sand dune curves in desert beige, carpet design with flowing linear waves, ornamental geometric pattern",
    "Tree ring pattern in forest brown, concentric circle motif for carpet design, decorative geometric rings, flat textile pattern",
    "Simplified coral branch pattern in ocean colors, geometric tree-like motif for textile design, stylized marine pattern",
    "Abstract cloud swirl pattern in sky white, decorative spiral motif for carpet design, geometric atmospheric curves",
    "Stylized lava flow lines in volcanic red, decorative branching pattern for rug design, geometric molten streams",
    "Ice crystal star motif in glacial blue, geometric angular pattern for carpet design, crystalline decorative shapes",
    "Mushroom gill radial pattern in forest earth, decorative spoke design for textile, geometric fan-like motif",
    "Seashell spiral pattern in pearl white, fibonacci curve motif for carpet design, decorative mathematical spiral",
    "Honeycomb hexagonal pattern in amber gold, regular geometric tessellation for carpet design, decorative cell structure",
    "Spider web radial pattern in morning silver, geometric spoke design for textile, decorative concentric circles with rays",
    "Bird nest weave pattern in natural brown, decorative interlaced lines for carpet design, geometric basket-weave motif",
    "Beaver dam log pattern in water brown, decorative parallel lines for rug design, geometric timber arrangement",
    "Ant tunnel network in earth red, abstract maze pattern for carpet design, geometric branching pathways",
    "Termite mound tower pattern in clay brown, decorative vertical lines for textile design, geometric architectural motif",
    "Stalactite icicle pattern in cave white, decorative pointed shapes for carpet design, geometric hanging formations",
    "Abstract erosion lines in canyon red, decorative branching pattern for rug design, geometric river-like curves",
    "Tide pool circle pattern in marine colors, decorative concentric rings for carpet design, geometric water motif",
    "Mountain ridge zigzag pattern in granite gray, decorative angular lines for textile design, geometric peak silhouettes",
    "River delta branch pattern in muddy brown, decorative tree-like design for carpet, geometric splitting lines",
    "Glacier crack pattern in ice blue, decorative angular lines for rug design, geometric fracture motif",
    "Desert crack polygon pattern in drought brown, decorative geometric shapes for carpet design, angular tile motif",
    "Forest canopy layer pattern in green gradient, decorative horizontal bands for textile design, geometric stratified design",
    "Ocean wave curve pattern in deep blue, decorative undulating lines for carpet design, geometric rhythmic waves",
    "Lightning branch pattern in electric white, decorative jagged lines for rug design, geometric forked tree motif",
    "Earthquake fault zigzag in geological brown, decorative angular breaks for carpet design, geometric crack pattern",
    "Volcanic ash scatter pattern in gray cloud, decorative speckled design for textile, geometric particle distribution",
    "Glacier flow curve pattern in ice white, decorative flowing lines for carpet design, geometric river-like bands",
    "Cave chamber circle pattern in stone gray, decorative oval shapes for rug design, geometric cavern motif",
    "Tidal wave curve pattern in coastal colors, decorative undulating design for carpet, geometric shore motif",
    "Wind carved spiral pattern in sandstone red, decorative swirl design for textile, geometric vortex motif",
    "River meander curve pattern in water blue, decorative S-curves for carpet design, geometric serpentine lines",
    "Crystal facet pattern in mineral colors, decorative angular shapes for rug design, geometric prism motif",
    "Soil layer stripe pattern in earth tones, decorative horizontal bands for carpet design, geometric sediment layers",
    "Root network branch pattern in underground brown, decorative tree design for textile, geometric spreading lines",
    "Mycelium web pattern in forest floor colors, decorative network design for carpet, geometric connection lines",
    "Blood vessel branch pattern in organic red, decorative tree-like design for rug, geometric circulatory motif",
    "Nerve pathway network in neural white, decorative branching lines for carpet design, geometric neural pattern",
    "Cell division circle pattern in microscopic colors, decorative splitting shapes for textile design, geometric mitosis motif",
    "DNA helix twist pattern in genetic blue, decorative spiral ribbon for carpet design, geometric double helix motif",
    "Protein fold pattern in biochemical colors, decorative curved lines for rug design, geometric molecular ribbons",
    "Membrane bubble pattern in cellular pink, decorative circle clusters for carpet design, geometric cell boundary motif",
    "Enzyme lock pattern in catalytic green, decorative interlocking shapes for textile design, geometric puzzle motif",
    "Chromosome X-pattern in genetic purple, decorative crossed lines for carpet design, geometric genetic motif",
    "Mitochondrial network in energy orange, decorative oval chains for rug design, geometric cellular organelle pattern",
    "Cytoskeleton grid pattern in structural gray, decorative mesh design for carpet, geometric cellular framework",
    "Vesicle dot pattern in cellular yellow, decorative circle clusters for textile design, geometric transport motif",
    "Nuclear pore circle pattern in atomic blue, decorative ring design for carpet, geometric gateway motif",
    "Ribosome cluster pattern in protein brown, decorative grouped circles for rug design, geometric manufacturing motif"
  ],
  "Traditional_Rug": [
    "Persian Isfahan medallion in royal blue and gold, classical central motif with hunting scenes and Safavid dynasty heritage",
    "Turkish Hereke silk prayer design in mihrab pattern, Islamic devotional with Mecca orientation and Ottoman court tradition",
    "Caucasian Kazak geometric shields in tribal red, warrior protection symbols with mountain clan heritage and nomadic strength",
    "Indian Agra Mughal garden in paradise layout, four-river pattern with imperial court tradition and Persian influence synthesis",
    "Moroccan Beni Ouarain diamond lattice in natural tone, Berber tribal pattern with Atlas Mountain heritage and nomadic simplicity",
    "Afghan Baluch prayer design in deep burgundy, Islamic devotional with tribal interpretation and nomadic religious practice",
    "Chinese Peking imperial dragon in golden yellow, celestial creature with Forbidden City tradition and dynastic power symbolism",
    "Tibetan tiger pattern in monastery colors, Buddhist tantric with Himalayan spiritual tradition and protective symbolism",
    "Kurdish tribal runner in earth tones, village pattern with Zagros Mountain heritage and pastoral nomadic life",
    "Turkmen Tekke main design in traditional red, nomadic tent decoration with Central Asian heritage and tribal identity",
    "Russian Karabagh hunting pattern in forest green, Caucasian noble with aristocratic hunting tradition and mountain heritage",
    "Armenian Karabagh garden in jewel tones, Christian monastery pattern with Armenian cultural heritage and religious symbolism",
    "Azerbaijani Shirvan prayer design in mosque blue, Islamic devotional with Caucasian regional interpretation and Sufi mysticism",
    "Georgian Bordjalou kazak in warrior colors, Caucasian tribal pattern with Christian-Islamic synthesis and mountain clan tradition",
    "Daghestan prayer design in calligraphic pattern, Islamic devotional with Arabic script and North Caucasus Muslim heritage",
    "Uzbek suzani embroidery in bride's colors, Central Asian wedding pattern with silk road heritage and matrimonial blessing",
    "Tajik felt design in nomadic pattern, mountain pastoral with Pamir heritage and high altitude adaptation",
    "Kyrgyz shyrdak felt in ancestral patterns, nomadic floor covering with Tian Shan heritage and horse culture tradition",
    "Kazakh felt design in steppe motif, nomadic dwelling pattern with Eurasian heritage and pastoral wandering tradition",
    "Mongolian pattern in ger tent colors, nomadic dwelling with grassland heritage and traditional yurt decoration",
    "Nepalese Tibetan meditation design in monastery red, Buddhist practice pattern with Himalayan heritage and spiritual contemplation",
    "Bhutanese pattern in Thunder Dragon colors, Himalayan kingdom with Buddhist heritage and mountain isolation tradition",
    "Pakistani Bokhara in traditional red, Turkmen revival pattern with subcontinental adaptation and Islamic cultural synthesis",
    "Indian dhurrie flat weave in village colors, floor covering with rural heritage and agricultural community tradition",
    "Rajasthani camel caravan in desert colors, merchant trade pattern with Thar Desert heritage and trading route tradition",
    "Kashmiri chain stitch in paradise design, Mughal garden pattern with vale heritage and Persian cultural influence",
    "Bengali kantha embroidery in recycled cotton pattern, rural women's with delta heritage and sustainable tradition",
    "South Indian temple pattern in devotional colors, Hindu sacred with Dravidian heritage and temple ritual tradition",
    "Syrian Aleppo room design in merchant colors, urban pattern with silk road heritage and commercial prosperity tradition",
    "Lebanese mountain pattern in cedar colors, Levantine highland with Phoenician heritage and Mediterranean tradition",
    "Jordanian Bedouin tent design in desert colors, nomadic pattern with Arabian Peninsula heritage and pastoral tradition",
    "Egyptian Coptic pattern in Nile colors, Christian with pharaonic synthesis and ancient civilization continuity",
    "Sudanese prayer design in Nubian colors, Islamic devotional with African synthesis and Nile valley cultural tradition",
    "Moroccan Middle Atlas in geometric red, Berber mountain pattern with traditional dyeing and Atlas highland heritage",
    "Tunisian kilim in Mediterranean colors, North African pattern with Carthaginian heritage and coastal trading tradition",
    "Algerian tribal pattern in Sahara colors, Berber nomadic with desert heritage and Tuareg cultural influence",
    "Ethiopian church pattern in Orthodox colors, Coptic Christian with highland heritage and ancient church tradition",
    "Senegalese ritual pattern in ceremonial colors, West African with Wolof heritage and Islamic-animist synthesis",
    "Malian mud cloth in earth pigment, Bambara tribal pattern with Sahel heritage and traditional dyeing technology",
    "Nigerian Hausa embroidery in royal colors, West African pattern with emirate heritage and Islamic calligraphic tradition",
    "Ghanaian kente in royal gold, Akan ceremonial pattern with Gold Coast heritage and traditional excellence",
    "Kenyan Maasai pattern in warrior colors, East African pastoral with savanna heritage and age-grade tradition",
    "Tanzania Kanga in Swahili proverb colors, coastal pattern with Indian Ocean heritage and linguistic cultural tradition",
    "South African Ndebele in geometric colors, Bantu artistic pattern with highveld heritage and architectural painting tradition",
    "Botswana basket weave in Kalahari colors, San traditional pattern with desert heritage and hunter-gatherer adaptation",
    "Namibian Himba pattern in ochre red, pastoral nomadic with desert heritage and cattle culture tradition",
    "Zambian chitenge in copper colors, Central African pattern with mining heritage and trade route cultural exchange",
    "Zimbabwe shona pattern in granite colors, Bantu artistic with plateau heritage and stone carving tradition",
    "Mozambique capulana in Indian Ocean colors, coastal pattern with Portuguese colonial synthesis and maritime heritage",
    "Madagascar lambas in highland colors, Malagasy traditional pattern with island heritage and Austronesian-African synthesis",
    "Mauritian pattern in tropical colors, island with colonial synthesis and multicultural heritage blend"
  ],
  
  
}
    
    # Configuration
    config = {
        "output_folder": "/data/generated_carpets",
        "images_per_prompt": 10,  
        "seeds_per_style": 1,     # Number of different seeds per artistic style
        "artistic_styles": len(artistic_styles),  # Number of artistic styles (5)
        "guidance_scale": 7.0,    # Fixed strength for all prompts
        "image_size": (896, 1200),
        "num_inference_steps": 50,
        "base_negative_prompt": (
            "blurry, low quality, distorted, warped, photographic, realistic, "
            "photograph, 3D, dimensional, shadows, lighting effects, depth, "
            "perspective, people, faces, text, watermark, signature, frame, "
            "border, poor quality, pixelated, noisy, grainy, artifacts, "
            "carpet texture, fabric texture, weave, fibers, physical material, "
            "photography, studio lighting, product shot, cluttered, messy, "
            "wildlife photography, nature documentary, safari, zoo, animals in wild, "
            "natural habitat, realistic animals, photorealistic wildlife, "
            "animal photography, nature scene, landscape, outdoor scene, "
            "real animals, living creatures, naturalistic, documentary style, "
            "animal portrait, wildlife scene, environmental background, "
            "forest scene, jungle scene, savanna, natural environment, "
            "realistic fur, realistic feathers, realistic scales, "
            "flowers, floral, petals, blooms, blossoms, roses, tulips, daisies, "
            "botanical, garden, meadow, field of flowers, flower arrangements"

        )
    }
    
    # Create output directories
    os.makedirs(config["output_folder"], exist_ok=True)
    
    # Initialize generator
    generator = SD35CarpetGenerator()
    
    # Track generation statistics
    total_images = 0
    successful_generations = 0
    failed_generations = 0
    
    # Calculate total target images
    total_prompts = sum(len(class_prompts) for class_prompts in prompts_data.values())
    total_target_images = total_prompts * config["images_per_prompt"]
    
    print(f"Starting carpet generation for {len(prompts_data)} classes")
    print(f"Mode: {config['images_per_prompt']} images per prompt ({config['seeds_per_style']} seeds × {config['artistic_styles']} styles)")
    print(f"Total prompts: {total_prompts}")
    print(f"Total target images: {total_target_images}")
    print(f"Guidance scale: {config['guidance_scale']} (fixed)")
    
    # Process each class
    for class_name, class_prompts in prompts_data.items():
        print(f"\n{'='*60}")
        print(f"Processing class: {class_name.upper()}")
        print(f"Prompts in class: {len(class_prompts)}")
        print(f"Images per prompt: {config['images_per_prompt']} ({config['seeds_per_style']} seeds × {config['artistic_styles']} styles)")
        print(f"Total images for class: {len(class_prompts) * config['images_per_prompt']}")
        print(f"{'='*60}")
        
        # Create class directory
        class_dir = Path(config["output_folder"]) / class_name
        os.makedirs(class_dir, exist_ok=True)
        
        class_successful = 0
        class_failed = 0
        
        print(f"Using all {len(class_prompts)} prompts with {config['images_per_prompt']} images each")
        
        # Generate multiple images for each prompt using different artistic styles and seeds
        for prompt_idx, prompt in enumerate(class_prompts):
            print(f"\nProcessing prompt {prompt_idx + 1}/{len(class_prompts)} for {class_name}")
            print(f"Prompt preview: {prompt[:80]}...")
            
            # Generate images for each artistic style
            for style_idx, artistic_style in enumerate(artistic_styles):
                print(f"  Style {style_idx + 1}/{len(artistic_styles)}: {artistic_style[:50]}...")
                
                # Generate 2 images with different seeds for this style
                for seed_idx in range(config["seeds_per_style"]):
                    try:
                        # Generate random seed 
                        seed = random.randint(0, 2**32 - 1)
                        
                        # Add artistic style and seamless pattern design enhancements to prompt
                        enhanced_prompt = (
                            f"{prompt}, in this style {artistic_style}, seamless repeating pattern, tileable, "
                            "perfect repeat, no visible seams, flat design, "
                            "vector style, clean lines, sharp edges, "
                            "solid colors, high contrast, "
                            "no shadows, no 3D effects"
                        )
                        
                        print(f"    Seed {seed_idx + 1}/{config['seeds_per_style']} (seed: {seed}, strength: {config['guidance_scale']:.2f})")
                        
                        # Generate seamless pattern image
                        image, actual_seed = generator.generate_carpet_image(
                            prompt=enhanced_prompt,
                            negative_prompt=config["base_negative_prompt"],
                            width=config["image_size"][0],
                            height=config["image_size"][1],
                            guidance_scale=config["guidance_scale"],
                            num_inference_steps=config["num_inference_steps"],
                            seed=seed,
                            enable_seamless=True
                        )
                        
                        # Save image with prompt index, style index, seed index, and strength
                        filename = f"{class_name}_p{prompt_idx + 1:03d}_style{style_idx + 1:01d}_s{seed_idx + 1:02d}_seed{actual_seed}_str{config['guidance_scale']:.2f}_seamless.png"
                        image_path = class_dir / filename
                        image.save(image_path, "PNG", quality=95)
                        
                        print(f"      ✓ Saved: {filename}")
                        
                        class_successful += 1
                        successful_generations += 1
                        total_images += 1
                        
                        # Clear GPU memory
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"      ✗ Failed to generate style {style_idx + 1} seed {seed_idx + 1} for prompt {prompt_idx + 1} in {class_name}: {str(e)}")
                        class_failed += 1
                        failed_generations += 1
                        total_images += 1
                        continue
        
        print(f"\nClass {class_name} complete:")
        expected_images = len(class_prompts) * config["images_per_prompt"]
        print(f"  Successful: {class_successful}/{expected_images}")
        print(f"  Failed: {class_failed}")
        print(f"  Success rate: {(class_successful/(class_successful + class_failed)*100):.1f}%")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Overall success rate: {(successful_generations/total_images*100):.1f}%")
    print(f"Generation mode: {config['images_per_prompt']} images per prompt ({config['seeds_per_style']} seeds × {config['artistic_styles']} styles)")
    print(f"Total classes: {len(prompts_data)}")
    print(f"Total prompts processed: {total_prompts}")
    print(f"Guidance scale: {config['guidance_scale']} (fixed)")
    print(f"Output directory: {config['output_folder']}")
    
    # Create summary file
    summary = {
        "generation_config": config,
        "statistics": {
            "total_images_processed": total_images,
            "successful_generations": successful_generations,
            "failed_generations": failed_generations,
            "success_rate": successful_generations/total_images*100 if total_images > 0 else 0,
            "classes_processed": list(prompts_data.keys()),
            "total_prompts": total_prompts,
            "images_per_prompt": config["images_per_prompt"],
            "seeds_per_style": config["seeds_per_style"],
            "artistic_styles_count": config["artistic_styles"],
            "generation_mode": f"{config['images_per_prompt']}_images_per_prompt_with_{config['seeds_per_style']}_seeds_x_{config['artistic_styles']}_styles"
        },
        "model_info": {
            "model_id": MODEL_ID,
            "guidance_scale": config["guidance_scale"],
            "num_inference_steps": config["num_inference_steps"],
            "image_size": config["image_size"]
        },
        "artistic_styles": artistic_styles
    }
    
    summary_path = Path(config["output_folder"]) / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    # Local execution
    with app.run():
        generate_carpet_dataset.remote()