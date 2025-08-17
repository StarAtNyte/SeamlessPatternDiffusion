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

app = modal.App("carpet-prompt-generator", image=image)

# Model configuration
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16

# Volume for storing generated images
output_volume = modal.Volume.from_name("generated-carpet-tileable", create_if_missing=True)

# FLUX-adapted seamless generation functions
def asymmetricConv2DConvForward_circular(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """Circular padding for Conv2d layers - adapted for FLUX architecture"""
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
def make_seamless_flux(model):
    """Enable circular padding on all Conv2d layers in FLUX model"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Handle LoRA compatible layers if they exist (FLUX may use them)
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = asymmetricConv2DConvForward_circular.__get__(module, Conv2d)


# Sets the padding mode back to default on Conv2d - FLUX compatible
def disable_seamless_flux(model):
    """Disable circular padding and restore default Conv2d behavior"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Handle LoRA compatible layers if they exist
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = nn.Conv2d._conv_forward.__get__(module, Conv2d)


class FluxCarpetGenerator:
    def __init__(self):
        self.model_id = MODEL_ID
        self.pipe = None

    def initialize_pipeline(self):
        """Initialize Flux Dev pipeline."""
        if self.pipe is None:
            from diffusers import FluxPipeline
            
            print("Loading Flux Dev model...")
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE,
                use_safetensors=True
            )
            
            self.pipe.enable_attention_slicing()
            self.pipe.to("cuda")
            print("Flux Dev model loaded successfully!")

    def flux_diffusion_callback(self, pipe, step_index, timestep, callback_kwargs):
        """
        Callback for seamless pattern generation adapted for FLUX architecture.
        FLUX uses rectified flow, so we adapt the timing and approach.
        """
        # For FLUX, we apply seamless techniques in the last 20% of steps
        # FLUX.1-dev typically uses ~28 steps
        total_steps = pipe.num_inference_steps if hasattr(pipe, 'num_inference_steps') else 28
        late_stage_threshold = int(total_steps * 0.8)
        
        # Apply circular padding to transformer and VAE in late stages
        if step_index >= late_stage_threshold:
            # FLUX has transformer blocks and VAE
            if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                make_seamless_flux(pipe.transformer)
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                make_seamless_flux(pipe.vae)

        # Noise rolling for early stages (adapted for FLUX's flow matching)
        if step_index < late_stage_threshold:
            # FLUX uses latents, but tensor structure may differ
            if "latents" in callback_kwargs:
                latents = callback_kwargs["latents"]
                # Check latent tensor dimensions before rolling
                if len(latents.shape) >= 4:  # Should be [batch, channels, height, width]
                    # Smaller shifts for FLUX as it's more sensitive
                    shift_amount = min(16, max(4, latents.shape[-1] // 32))
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
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
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
        
        # Ensure seamless is disabled before starting (FLUX adaptation)
        if enable_seamless:
            if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                disable_seamless_flux(self.pipe.transformer)
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                disable_seamless_flux(self.pipe.vae)
        
        with torch.autocast("cuda", dtype=DTYPE):
            if enable_seamless:
                # Use FLUX seamless generation with callback
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=self.flux_diffusion_callback
                ).images[0]
            else:
                # Standard FLUX generation without seamless techniques
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
    volumes={"/data": output_volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_carpet_dataset():
    """Generate carpet dataset from prompts JSON file."""
    
    # Define artistic styles to prepend to prompts
    artistic_styles = [
        "Watercolor: soft translucent washes with organic bleeds",
        "Oil Paint: rich lustrous colors with visible brushstrokes",
        "Block Print: hand-carved aesthetic with bold graphic designs",
        "Ink Drawing: bold black lines with crosshatching stippling",
        "Digital Vector: clean scalable designs with smooth curves",
        "Batik: wax-resist technique with organic flowing patterns",
        "Screen Print: flat uniform colors with graphic aesthetic",
        "Hand-painted Brushwork: visible brushstrokes with artisanal irregularities",
        "Pencil Sketch: soft grayscale tones with subtle shading",
        "Gouache Illustration: opaque matte finish with saturated colors"
    ]
    
    # Test prompts optimized for FLUX seamless pattern generation
    prompts_data = {
  "Minimalistic": [
    "Single horizontal line in charcoal gray thread, minimal linear carpet element with Zen aesthetic and Japanese ma spatial concept",
    "Three circles in perfect alignment in neutral wool, geometric minimalism carpet with Bauhaus principle and modernist reduction philosophy",
    "Vertical stripe in navy blue thread, simple linear carpet pattern with Scandinavian design and functional aesthetic restraint",
    "Square grid in pale gray wool, minimal geometric carpet structure with Swiss design grid system and typographic precision",
    "Two-tone diagonal split in black and white thread, stark minimalism carpet with De Stijl movement and absolute contrast",
    "Dotted line pattern in subtle beige wool, understated repetition carpet with minimalist principle and quiet visual rhythm",
    "Single curve in soft white thread, organic minimalism carpet with modernist sculpture and essential form reduction",
    "Rectangle frame in light gray wool, structural minimalism carpet with architectural principle and spatial boundary definition",
    "Parallel lines in muted blue thread, linear minimalism carpet with modernist aesthetic and geometric purity",
    "Corner accent in warm gray wool, architectural minimalism carpet with building detail and structural accent point",
    "Circular void in neutral-toned thread, negative space minimalism carpet with spatial concept and absence as presence",
    "Thin border in pale silver wool, edge minimalism carpet with frame concept and boundary definition subtlety",
    "Triangle point in monochrome thread, angular minimalism carpet with directional element and geometric essence",
    "Gradient fade in soft gray wool, tonal minimalism carpet with atmospheric effect and color transition subtlety",
    "Cross intersection in minimal black thread, linear junction carpet with Swiss design and functional symbol reduction",
    "Wave line in light blue wool, organic minimalism carpet with natural form and essential movement capture",
    "Grid dot in neutral beige thread, structural minimalism carpet with measurement system and organizational principle",
    "Diagonal accent in charcoal wool, angular minimalism carpet with dynamic element and directional emphasis",
    "Circular outline in pale gold thread, geometric minimalism carpet with perfect form and essential shape definition",
    "Vertical division in soft white wool, spatial minimalism carpet with architectural proportion and golden ratio",
    "Minimal texture in stone gray thread, surface minimalism carpet with material essence and tactile suggestion",
    "Linear progression in gradient gray wool, sequential minimalism carpet with mathematical order and visual rhythm",
    "Corner radius in warm white thread, architectural minimalism carpet with building detail and modernist refinement",
    "Asymmetric balance in neutral-toned wool, compositional minimalism carpet with visual weight and spatial harmony",
    "Narrow stripe in pale blue thread, linear minimalism carpet with textile reference and functional decoration",
    "Rectangle sequence in monochrome wool, modular minimalism carpet with systematic order and geometric progression",
    "Subtle shadow in soft gray thread, atmospheric minimalism carpet with light effect and dimensional suggestion",
    "Edge detail in minimal silver wool, architectural minimalism carpet with building trim and structural precision",
    "Geometric intersection in pale gold thread, mathematical minimalism carpet with pure form and essential geometry",
    "Minimal border in neutral beige wool, frame minimalism carpet with boundary definition and spatial containment",
    "Clean line in charcoal gray thread, essential minimalism carpet with drawing reduction and linear purity",
    "Simple curve in soft white wool, organic minimalism carpet with natural form and movement essence",
    "Grid structure in light gray thread, organizational minimalism carpet with system logic and rational order",
    "Minimal accent in warm silver wool, decorative minimalism carpet with essential ornament and functional beauty",
    "Geometric void in neutral-toned thread, negative minimalism carpet with spatial concept and emptiness as design",
    "Linear rhythm in pale blue wool, sequential minimalism carpet with musical analogy and visual tempo",
    "Structural element in minimal black thread, architectural minimalism carpet with building component and functional design",
    "Tonal variation in soft gray wool, atmospheric minimalism carpet with color subtlety and perceptual refinement",
    "Essential form in warm white thread, sculptural minimalism carpet with pure shape and material honesty",
    "Minimal detail in light gold wool, decorative minimalism carpet with essential ornament and luxury restraint",
    "Geometric progression in neutral sequence thread, mathematical minimalism carpet with logical order and visual calculation",
    "Spatial division in pale silver wool, architectural minimalism carpet with proportion system and golden section",
    "Clean geometry in monochrome thread, pure minimalism carpet with essential form and absolute reduction",
    "Minimal texture in stone beige wool, material minimalism carpet with surface quality and tactile suggestion",
    "Linear sequence in soft gray thread, rhythmic minimalism carpet with visual music and geometric tempo",
    "Essential detail in warm charcoal wool, architectural minimalism carpet with building accent and structural poetry",
    "Geometric purity in neutral white thread, absolute minimalism carpet with perfect form and essential geometry",
    "Minimal rhythm in pale blue sequence wool, musical minimalism carpet with visual tempo and systematic beauty",
    "Structural poetry in light gray thread, architectural minimalism carpet with building essence and spatial meditation",
    "Essential beauty in minimal gold wool, luxury minimalism carpet with precious restraint and quiet elegance"
  ],
  "Organic": [
    "Flowing water ripples in aquatic blue thread, liquid movement carpet pattern with stream dynamics and natural fluid motion weave",
    "Smooth river stones in earth-toned wool, geological carpet pattern with water erosion and natural weathering processes texture",
    "Wind-carved sand dunes in desert beige thread, aeolian formation carpet with atmospheric sculpting and arid landscape dynamics",
    "Tree bark texture in forest brown wool, botanical surface carpet with natural growth rings and organic aging processes pattern",
    "Coral reef formations in ocean-colored thread, marine organism carpet with calcium carbonate and underwater ecosystem structure",
    "Cloud formations in sky white wool, atmospheric carpet pattern with condensation dynamics and weather system movement",
    "Lava flow patterns in volcanic red thread, molten rock carpet with geological formation and igneous cooling processes texture",
    "Ice crystal formations in glacial blue wool, frozen water carpet with crystalline structure and winter weathering patterns",
    "Mushroom gill patterns in forest earth thread, fungal structure carpet with spore distribution and decomposition ecosystem role",
    "Seashell spiral chambers in pearl white wool, mollusk architecture carpet with fibonacci growth and marine adaptation pattern",
    "Honeycomb cell structure in amber gold thread, insect architecture carpet with hexagonal efficiency and natural engineering",
    "Spider web radial pattern in morning silver wool, arachnid engineering carpet with protein fiber and predatory adaptation",
    "Bird nest construction in natural brown thread, avian architecture carpet with material weaving and protective structure",
    "Beaver dam engineering in water brown wool, mammal construction carpet with environmental modification and aquatic habitat",
    "Ant hill tunnel system in earth red thread, insect architecture carpet with social organization and underground city planning",
    "Termite mound ventilation in clay brown wool, insect engineering carpet with temperature control and architectural sophistication",
    "Stalactite formations in cave white thread, geological dripping carpet with mineral deposition and underground sculpture",
    "Erosion patterns in canyon red wool, water carving carpet with geological time and landscape formation dynamics",
    "Tide pool ecosystem in marine-colored thread, intertidal zone carpet with species adaptation and coastal habitat complexity",
    "Mountain ridge formation in granite gray wool, geological uplift carpet with tectonic forces and alpine landscape creation",
    "River delta branching in muddy brown thread, sediment deposition carpet with water distribution and coastal land formation",
    "Glacier crevasse patterns in ice blue wool, frozen movement carpet with pressure dynamics and polar landscape features",
    "Desert cracking patterns in drought brown thread, moisture loss carpet with soil contraction and arid environment adaptation",
    "Forest canopy layers in green gradient wool, woodland structure carpet with light filtration and vertical ecosystem zones",
    "Ocean wave mechanics in deep blue thread, fluid dynamics carpet with energy transfer and coastal interaction patterns",
    "Lightning branch patterns in electric white wool, atmospheric discharge carpet with electrical pathways and storm energy",
    "Earthquake fault lines in geological brown thread, tectonic movement carpet with crustal fracture and seismic energy release",
    "Volcanic ash dispersal in gray cloud wool, pyroclastic flow carpet with atmospheric distribution and geological impact",
    "Glacier movement patterns in ice white thread, frozen flow carpet with valley carving and landscape sculpting forces",
    "Underground cave systems in stone gray wool, water dissolution carpet with limestone carving and subterranean architecture",
    "Tidal erosion patterns in coastal-colored thread, wave action carpet with shoreline modification and marine landscape dynamics",
    "Wind erosion formations in sandstone red wool, atmospheric carving carpet with rock sculpture and desert landscape creation",
    "River meander patterns in water blue thread, fluid dynamics carpet with landscape carving and geographical evolution",
    "Crystal growth formations in mineral-colored wool, atomic arrangement carpet with geometric structure and geological beauty",
    "Soil stratification layers in earth-toned thread, geological history carpet with sediment deposition and temporal landscape record",
    "Root system networks in underground brown wool, botanical architecture carpet with nutrient distribution and soil stabilization",
    "Mycelium thread networks in forest floor thread, fungal connection carpet with nutrient exchange and woodland communication system",
    "Blood vessel branching in organic red wool, circulatory pattern carpet with fluid distribution and biological transportation",
    "Nerve pathway networks in neural white thread, biological communication carpet with electrical signaling and information transfer",
    "Cell division patterns in microscopic-colored wool, biological reproduction carpet with genetic replication and life multiplication",
    "DNA helix structure in genetic blue thread, molecular architecture carpet with information storage and biological blueprint",
    "Protein folding patterns in biochemical-colored wool, molecular origami carpet with functional structure and biological machinery",
    "Membrane surface textures in cellular pink thread, biological boundary carpet with selective permeability and life protection",
    "Enzyme active sites in catalytic green wool, biochemical precision carpet with molecular recognition and biological efficiency",
    "Chromosome condensation in genetic purple thread, biological organization carpet with information packaging and cellular division",
    "Mitochondrial networks in energy orange wool, cellular powerhouse carpet with metabolic function and biological energy production",
    "Cytoskeleton frameworks in structural gray thread, cellular architecture carpet with mechanical support and biological scaffolding",
    "Vesicle transport systems in cellular yellow wool, biological logistics carpet with molecular delivery and cellular communication",
    "Nuclear pore complexes in atomic blue thread, cellular gateways carpet with selective transport and biological security",
    "Ribosome assembly patterns in protein brown wool, biological machinery carpet with genetic translation and molecular manufacturing"
  ],
  "Traditional_Rug": [
    "Persian Isfahan medallion in royal blue and gold wool, classical central motif hand-knotted with hunting scenes and Safavid dynasty heritage carpet",
    "Turkish Hereke silk prayer rug in mihrab design, Islamic devotional pattern hand-knotted with Mecca orientation and Ottoman court tradition",
    "Caucasian Kazak geometric shields in tribal red wool, warrior protection symbols hand-knotted with mountain clan heritage and nomadic strength",
    "Indian Agra Mughal garden in paradise layout wool, four-river pattern hand-knotted with imperial court tradition and Persian influence synthesis",
    "Moroccan Beni Ouarain diamond lattice in natural wool, Berber tribal pattern hand-knotted with Atlas Mountain heritage and nomadic simplicity",
    "Afghan Baluch prayer rug in deep burgundy wool, Islamic devotional hand-knotted with tribal interpretation and nomadic religious practice",
    "Chinese Peking imperial dragon in golden yellow silk, celestial creature hand-knotted with Forbidden City tradition and dynastic power symbolism",
    "Tibetan tiger rug in monastery-colored wool, Buddhist tantric pattern hand-knotted with Himalayan spiritual tradition and protective symbolism",
    "Kurdish tribal runner in earth-toned wool, village weaving hand-knotted with Zagros Mountain heritage and pastoral nomadic life",
    "Turkmen Tekke main carpet in traditional red wool, nomadic tent decoration hand-knotted with Central Asian heritage and tribal identity",
    "Russian Karabagh hunting carpet in forest green wool, Caucasian noble pattern hand-knotted with aristocratic hunting tradition and mountain heritage",
    "Armenian Karabagh garden in jewel-toned wool, Christian monastery pattern hand-knotted with Armenian cultural heritage and religious symbolism",
    "Azerbaijani Shirvan prayer rug in mosque blue wool, Islamic devotional hand-knotted with Caucasian regional interpretation and Sufi mysticism",
    "Georgian Bordjalou kazak in warrior-colored wool, Caucasian tribal pattern hand-knotted with Christian-Islamic synthesis and mountain clan tradition",
    "Daghestan prayer rug in calligraphic design wool, Islamic devotional hand-knotted with Arabic script and North Caucasus Muslim heritage",
    "Uzbek suzani embroidery in bride's colored silk, Central Asian wedding pattern hand-embroidered with silk road heritage and matrimonial blessing",
    "Tajik felt carpet in nomadic design wool, mountain pastoral pattern hand-felted with Pamir heritage and high altitude adaptation",
    "Kyrgyz shyrdak felt in ancestral patterns wool, nomadic floor covering hand-felted with Tian Shan heritage and horse culture tradition",
    "Kazakh felt carpet in steppe motif wool, nomadic dwelling pattern hand-felted with Eurasian heritage and pastoral wandering tradition",
    "Mongolian rug in ger tent-colored wool, nomadic dwelling pattern hand-knotted with grassland heritage and traditional yurt decoration",
    "Nepalese Tibetan meditation rug in monastery red wool, Buddhist practice pattern hand-knotted with Himalayan heritage and spiritual contemplation",
    "Bhutanese textile in Thunder Dragon-colored wool, Himalayan kingdom pattern hand-woven with Buddhist heritage and mountain isolation tradition",
    "Pakistani Bokhara in traditional red wool, Turkmen revival pattern hand-knotted with subcontinental adaptation and Islamic cultural synthesis",
    "Indian dhurrie flat weave in village-colored cotton, floor covering hand-woven with rural heritage and agricultural community tradition",
    "Rajasthani camel caravan in desert-colored wool, merchant trade pattern hand-woven with Thar Desert heritage and trading route tradition",
    "Kashmiri chain stitch in paradise design wool, Mughal garden pattern hand-embroidered with vale heritage and Persian cultural influence",
    "Bengali kantha embroidery in recycled cotton, rural women's pattern hand-stitched with delta heritage and sustainable textile tradition",
    "South Indian temple carpet in devotional-colored silk, Hindu sacred pattern hand-woven with Dravidian heritage and temple ritual tradition",
    "Syrian Aleppo room carpet in merchant-colored wool, urban pattern hand-knotted with silk road heritage and commercial prosperity tradition",
    "Lebanese mountain weaving in cedar-colored wool, Levantine highland pattern hand-woven with Phoenician heritage and Mediterranean tradition",
    "Jordanian Bedouin tent carpet in desert-colored wool, nomadic pattern hand-woven with Arabian Peninsula heritage and pastoral tradition",
    "Egyptian Coptic textile in Nile-colored linen, Christian pattern hand-woven with pharaonic synthesis and ancient civilization continuity",
    "Sudanese prayer rug in Nubian-colored wool, Islamic devotional hand-knotted with African synthesis and Nile valley cultural tradition",
    "Moroccan Middle Atlas in geometric red wool, Berber mountain pattern hand-woven with traditional dyeing and Atlas highland heritage",
    "Tunisian kilim in Mediterranean-colored wool, North African pattern hand-woven with Carthaginian heritage and coastal trading tradition",
    "Algerian tribal weaving in Sahara-colored wool, Berber nomadic pattern hand-woven with desert heritage and Tuareg cultural influence",
    "Ethiopian church textile in Orthodox-colored cotton, Coptic Christian pattern hand-woven with highland heritage and ancient church tradition",
    "Senegalese ritual textile in ceremonial-colored cotton, West African pattern hand-woven with Wolof heritage and Islamic-animist synthesis",
    "Malian mud cloth in earth pigment cotton, Bambara tribal pattern hand-painted with Sahel heritage and traditional dyeing technology",
    "Nigerian Hausa embroidery in royal-colored cotton, West African pattern hand-embroidered with emirate heritage and Islamic calligraphic tradition",
    "Ghanaian kente in royal gold silk, Akan ceremonial pattern hand-woven with Gold Coast heritage and traditional weaving excellence",
    "Kenyan Maasai textile in warrior-colored cotton, East African pastoral pattern hand-woven with savanna heritage and age-grade tradition",
    "Tanzania Kanga in Swahili proverb cotton, coastal pattern hand-printed with Indian Ocean heritage and linguistic cultural tradition",
    "South African Ndebele in geometric-colored cotton, Bantu artistic pattern hand-painted with highveld heritage and architectural painting tradition",
    "Botswana basket weave in Kalahari-colored grass, San traditional pattern hand-woven with desert heritage and hunter-gatherer adaptation",
    "Namibian Himba textile in ochre red leather, pastoral nomadic pattern hand-treated with desert heritage and cattle culture tradition",
    "Zambian chitenge in copper-colored cotton, Central African pattern hand-printed with mining heritage and trade route cultural exchange",
    "Zimbabwe shona textile in granite-colored cotton, Bantu artistic pattern hand-woven with plateau heritage and stone carving tradition",
    "Mozambique capulana in Indian Ocean-colored cotton, coastal pattern hand-printed with Portuguese colonial synthesis and maritime heritage",
    "Madagascar lambas in highland-colored silk, Malagasy traditional pattern hand-woven with island heritage and Austronesian-African synthesis",
    "Mauritian textile in tropical-colored cotton, island pattern hand-woven with colonial synthesis and multicultural heritage blend"
  ]
}
    
    # Configuration
    config = {
        "output_folder": "/data/generated_carpets",
        "images_per_prompt": 10,  
        "seeds_per_style": 1,     # Number of different seeds per artistic style
        "artistic_styles": len(artistic_styles),  # Number of artistic styles (5)
        "guidance_scale": 3.5,    # Fixed strength for all prompts
        "image_size": (896, 1200),
        "num_inference_steps": 35,
        "base_negative_prompt": (
            "blurry, low quality, distorted, warped, photographic, realistic, "
            "photograph, 3D, dimensional, shadows, lighting effects, depth, "
            "flowers, floral, petals, blooms, blossoms, roses, tulips, daisies, "
            "botanical, garden, meadow, field of flowers, flower heads, flower buds, "
            "flowering plants, flowering vines, floral patterns, floral motifs, "
            "flower arrangements, bouquets, corsages, garlands, wreaths, "
            "lily, orchid, sunflower, peony, carnation, iris, hibiscus, jasmine, "
            "cherry blossom, sakura, lotus flower, water lily, poppy, lavender, "
            "marigold, chrysanthemum, magnolia, azalea, camellia, gardenia, "
            "flower garden, botanical garden, greenhouse, nursery, floriculture"
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

        )
    }
    
    # Create output directories
    os.makedirs(config["output_folder"], exist_ok=True)
    
    # Initialize generator
    generator = FluxCarpetGenerator()
    
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
                            f"{prompt}, {artistic_style}, seamless repeating pattern design, tile-able, "
                            "perfect pattern repeat, no visible seams, continuous design, "
                            "pattern tile, wallpaper design, textile pattern, flat design, "
                            "vector art style, graphic design, clean lines, sharp edges, "
                            "solid colors, high contrast, decorative motif, "
                            "no shadows, no 3D effects, flat illustration"
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