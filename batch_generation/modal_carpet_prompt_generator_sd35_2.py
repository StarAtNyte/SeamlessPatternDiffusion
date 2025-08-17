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
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DTYPE = torch.bfloat16

# Volumes
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
        """Initialize SD3.5 pipeline."""
        if self.pipe is None:
            from diffusers import StableDiffusion3Pipeline
            
            print("Loading SD3.5 model...")
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE,
                use_safetensors=True
            )
            
            self.pipe.enable_attention_slicing()
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
  "Border": [
    "Greek key meander border with traditional carpet border proportions, interlocking rectangular spirals in cream on navy ground, continuous geometric repeat",
    "Celtic knotwork border with proportionally balanced frame, interwoven rope pattern in emerald on beige, corner variations complementing main field",
    "Art Deco sunburst border as narrow decorative frame, radiating lines and stepped corners in gold on black, angular geometric design",
    "Byzantine mosaic border with wide ornamental proportions, tessellated diamond patterns in jewel tones on ivory ground, rich color gradations",
    "Egyptian hieroglyph border with traditional carpet border proportions, stylized lotus and papyrus motifs in sandstone colors, alternating pattern sequence",
    "Chinese cloud scroll border with proportionally balanced frame, flowing S-curves in imperial yellow on crimson, continuous wave motion",
    "Islamic geometric star border with wide ornamental proportions, 8-pointed stars and interlacing polygons in turquoise on white ground",
    "Art Nouveau vine border that complements the main field, sinuous botanical stems and stylized leaves in sage green, organic flowing lines",
    "Persian palmette border with traditional carpet border proportions, stylized fan-shaped leaves in royal blue on cream, classic carpet motifs",
    "Roman acanthus border that complements the main field, carved leaf scrollwork in terra cotta on beige, classical architectural detail",
    "Japanese wave border as narrow decorative frame, stylized water crests in indigo blue on white, repeating ocean pattern",
    "Indian paisley border with proportionally balanced frame, teardrop motifs and dot filling in saffron on burgundy, traditional Kashmir design",
    "Gothic trefoil border that complements the main field, three-leaf arch patterns in gray on cream, medieval architectural motif",
    "Russian folk border as narrow decorative frame, simplified flower rosettes in bright red on white, peasant art style",
    "Moorish tile border with wide ornamental proportions, geometric star and cross patterns in azure blue on white, mathematical precision",
    "Viking knotwork border with traditional carpet border proportions, interlaced animal forms in iron gray on beige, Norse artistic tradition",
    "Aztec step-fret border with proportionally balanced frame, stepped pyramid motifs in black and white, pre-Columbian geometric design",
    "Turkish tulip border that complements the main field, stylized flower heads in red on cream, Ottoman court garden motif",
    "French fleur-de-lis border as narrow decorative frame, heraldic lily motifs in purple on gold, royal French symbolism",
    "Scottish tartan border with wide ornamental proportions, plaid stripe pattern in clan colors, Highland textile tradition",
    "Chevron zigzag border as narrow decorative frame, sharp V-shaped lines in navy and white stripes, bold directional pattern",
    "Diamond lattice border with traditional carpet border proportions, connected rhombus shapes in burgundy on cream, geometric network design",
    "Rope twist border as delicate narrow frame, braided cable pattern in natural hemp color, nautical-inspired texture",
    "Scalloped shell border with proportionally balanced frame, overlapping fan shapes in pearl gray on white, coastal decorative motif",
    "Herringbone border that complements the main field, angled brick pattern in earth tones, woven textile structure effect",
    "Coin dot border as narrow decorative frame, circular medallions in gold on navy, repeated geometric accent pattern",
    "Sawtooth border as delicate narrow frame, triangular teeth pattern in contrasting colors, sharp geometric edge design",
    "Guilloche border with proportionally balanced frame, interlaced circular bands in silver on black, classical architectural ornament",
    "Fretwork border with traditional carpet border proportions, angular maze pattern in burgundy on beige, geometric puzzle design",
    "Pearl strand border as narrow decorative frame, connected circular beads in ivory on gray, delicate jewelry-inspired pattern",
    "Laurel leaf border that complements the main field, victory wreath pattern in olive green on cream, classical Roman symbolism",
    "Dentil border as delicate narrow frame, rectangular tooth pattern in white on navy, architectural molding detail",
    "Egg and dart border with proportionally balanced frame, oval and arrow alternating pattern in stone colors, classical ornament",
    "Bead and reel border as narrow decorative frame, circular and cylindrical alternating shapes, architectural string course",
    "Cable border that complements the main field, twisted rope pattern in natural fiber colors, maritime decorative element",
    "Spiral wave border with proportionally balanced frame, continuous S-curve pattern in ocean blue on white, flowing water motion",
    "Star and dot border as narrow decorative frame, alternating stellar and circular shapes in gold on burgundy",
    "Triangle and square border with traditional carpet border proportions, alternating geometric shapes in contrasting colors, basic form pattern",
    "Interlocked rings border with proportionally balanced frame, overlapping circles in metallic tones, chain-link inspired design",
    "Arrow flight border as narrow decorative frame, pointed directional shapes in earth tones, Native American inspired",
    "Flower and stem border that complements the main field, simplified botanical motifs in spring colors, garden-inspired pattern",
    "Cross and circle border with proportionally balanced frame, alternating religious and secular symbols in traditional colors",
    "Hexagon honeycomb border with traditional carpet border proportions, tessellated six-sided shapes in honey gold, natural geometric pattern",
    "Lightning bolt border as narrow decorative frame, zigzag electrical pattern in electric blue on dark ground",
    "Feather edge border with proportionally balanced frame, stylized plume pattern in natural colors, bird-inspired decorative motif",
    "Ribbon twist border that complements the main field, curved band pattern in silk-like colors, textile-inspired flowing design",
    "Seed pod border as narrow decorative frame, oval botanical shapes in autumn colors, natural harvest motif",
    "Compass rose border with wide ornamental proportions, directional star pattern in navy and gold, nautical navigation symbol",
    "Scales border with traditional carpet border proportions, overlapping curved shapes in silver on blue, fish or armor inspired pattern",
    "Pine cone border with proportionally balanced frame, stylized conifer seed pattern in forest colors, woodland decorative motif"
  ],
  "Mandala": [
    "Tibetan sand mandala-inspired carpet in saffron and crimson wool, sacred Buddhist pattern hand-knotted with impermanence teaching symbolism",
    "Hindu lotus mandala in spiritual gold thread, eight-petaled sacred geometry woven with chakra symbolism and Vedic cosmic representation",
    "Zen circle mandala in black wool minimalism, ensō-inspired carpet pattern hand-knotted with Buddhist emptiness teaching and Japanese aesthetic",
    "Celtic spiral mandala in ancient green thread, triple spiral pattern woven with druidic symbolism and Irish cultural spiritual heritage",
    "Native American medicine wheel in earth-toned wool, four directions mandala hand-knotted with tribal wisdom and sacred hoop spiritual teaching",
    "Islamic geometric mandala in mosque blue thread, mathematical pattern woven with divine unity symbolism and Sufi spiritual contemplation",
    "Aztec calendar mandala in solar gold wool, cosmic pattern hand-knotted with Mesoamerican astronomy and indigenous time-keeping knowledge",
    "Egyptian ankh mandala in pharaonic gold thread, life symbol pattern woven with ancient Egyptian spirituality and Nile valley sacred geometry",
    "Greek labyrinth mandala in marble white wool, maze pattern hand-knotted with ancient mystery school tradition and Mediterranean spiritual journey",
    "Persian garden mandala in paradise green thread, four-garden pattern woven with Islamic paradise symbolism and Middle Eastern garden design",
    "Chinese yin-yang mandala in cosmic black and white wool, Taoist pattern hand-knotted with balance philosophy and ancient Chinese spiritual harmony",
    "Japanese chrysanthemum mandala in imperial purple thread, sixteen-petal pattern woven with imperial symbolism and Shinto spiritual reverence",
    "Indian rangoli mandala in festival-colored wool, floor pattern-inspired carpet hand-knotted with Hindu celebration tradition and spiritual protection",
    "Thai temple mandala in golden bronze thread, Buddhist architecture pattern woven with Southeast Asian temple design and Theravada meditation",
    "Cambodian Angkor mandala in temple stone-colored wool, Khmer pattern hand-knotted with Hindu-Buddhist synthesis and ancient kingdom grandeur",
    "Burmese pagoda mandala in lacquer red thread, stupa pattern woven with Theravada Buddhism and Myanmar cultural spiritual devotion",
    "Sri Lankan dagoba mandala in white limestone wool, Buddhist reliquary pattern hand-knotted with island Buddhism and Sinhalese heritage",
    "Nepalese stupa mandala in prayer flag-colored thread, Himalayan pattern woven with Tibetan Buddhism and mountain kingdom spiritual elevation",
    "Bhutanese dzong mandala in fortress white wool, monastery pattern hand-knotted with Himalayan Buddhism and Thunder Dragon kingdom protection",
    "Mongolian eternal knot mandala in nomad blue thread, endless pattern woven with Tibetan Buddhism and steppe cultural spiritual continuity",
    "Siberian shaman mandala in arctic white wool, spirit journey pattern hand-knotted with indigenous shamanism and polar cultural spiritual vision",
    "Australian Aboriginal mandala in ochre earth thread, dreamtime pattern woven with indigenous spirituality and outback cultural sacred geography",
    "Maori spiral mandala in jade green wool, koru pattern hand-knotted with Polynesian spirituality and New Zealand cultural life force symbolism",
    "Hawaiian lei mandala in tropical-colored thread, flower circle pattern woven with Polynesian aloha spirit and Pacific island cultural celebration",
    "Inuit snow mandala in polar white wool, arctic pattern hand-knotted with indigenous spirituality and northern cultural seasonal celebration",
    "African shield mandala in tribal-colored thread, protection pattern woven with indigenous spirituality and continental cultural ancestral wisdom",
    "Mayan cosmic mandala in jungle green wool, calendar pattern hand-knotted with Mesoamerican astronomy and ancient Maya spiritual cycles",
    "Inca sun mandala in Andean gold thread, Inti pattern woven with indigenous Andean spirituality and mountain cultural solar reverence",
    "Cherokee medicine mandala in forest-colored wool, healing pattern hand-knotted with Native American spirituality and woodland cultural harmony",
    "Lakota sacred hoop mandala in plains-colored wool, four directions pattern hand-knotted with Great Plains spirituality and nomadic cultural sacred movement",
    "Hopi spiral mandala in desert-colored thread, emergence pattern woven with Pueblo spirituality and southwestern cultural seasonal ceremony",
    "Navajo sand painting mandala in earth pigment-colored wool, healing pattern hand-knotted with Diné spirituality and desert cultural temporary sacred art",
    "Apache four directions mandala in mountain-colored thread, directional pattern woven with southwestern spirituality and nomadic cultural spiritual orientation",
    "Pueblo sun symbol mandala in adobe-colored wool, solar pattern hand-knotted with Rio Grande spirituality and agricultural cultural seasonal reverence",
    "Ojibwe medicine wheel mandala in Great Lakes-colored thread, seasonal pattern woven with woodland spirituality and northern cultural natural harmony",
    "Iroquois tree of life mandala in forest green wool, sacred tree pattern hand-knotted with longhouse spirituality and northeastern cultural cosmic connection",
    "Seminole patchwork mandala in swamp-colored thread, geometric pattern woven with southeastern spirituality and wetland cultural artistic tradition",
    "Tlingit raven mandala in Pacific-colored wool, creator pattern hand-knotted with Pacific Northwest spirituality and coastal cultural totem wisdom",
    "Haida eagle mandala in cedar-colored thread, thunderbird pattern woven with Pacific Northwest spirituality and island cultural carved tradition",
    "Kwakwaka'wakw sun mandala in potlatch-colored wool, solar pattern hand-knotted with Pacific Northwest spirituality and coastal cultural ceremonial power",
    "Blackfoot buffalo mandala in prairie-colored thread, sacred animal pattern woven with Great Plains spirituality and nomadic cultural life source reverence",
    "Cree star mandala in northern-colored wool, celestial pattern hand-knotted with subarctic spirituality and boreal cultural navigation wisdom",
    "Métis flower mandala in mixed heritage-colored thread, beadwork pattern woven with blended spirituality and prairie cultural artistic synthesis",
    "Innu caribou mandala in tundra-colored wool, migration pattern hand-knotted with subarctic spirituality and nomadic cultural animal spirit connection",
    "Mi'kmaq sunrise mandala in maritime-colored thread, dawn pattern woven with Atlantic coastal spirituality and maritime cultural daily renewal",
    "Haudenosaunee clan mandala in longhouse-colored wool, kinship pattern hand-knotted with Iroquoian spirituality and northeastern cultural social harmony",
    "Anishinaabe wild rice mandala in lake-colored thread, harvest pattern woven with Great Lakes spirituality and woodland cultural seasonal abundance",
    "Dakota sacred pipe mandala in ceremonial-colored wool, prayer pattern hand-knotted with plains spirituality and river cultural sacred communication",
    "Potawatomi fire mandala in council-colored thread, sacred flame pattern woven with Great Lakes spirituality and woodland cultural community gathering",
    "Menominee wild rice mandala in river-colored wool, water grain pattern hand-knotted with Great Lakes spirituality and woodland cultural sustainable harvest",
    "Winnebago earth lodge mandala in prairie-colored thread, dwelling pattern woven with Great Lakes spirituality and woodland cultural cosmic home"
  ],
  "Mirror": [
    "Butterfly wing symmetry design in iridescent blue, perfect bilateral reflection pattern with lepidoptera wing-inspired design",
    "Rorschach inkblot reflection pattern in deep black, psychological symmetry design with mirror inkblot-inspired design",
    "Kaleidoscope crystal reflection design in rainbow prism, multi-axis symmetry pattern with optical instrument-inspired geometry",
    "Snowflake hexagonal symmetry pattern in ice crystal white, six-fold reflection design with winter precipitation-inspired design",
    "Peacock feather eye reflection design in emerald green, radial symmetry pattern with bird plumage-inspired design",
    "Mandala mirror meditation pattern in spiritual gold, contemplative symmetry design with Buddhist reflection-inspired geometry",
    "Art Deco fan reflection design in metallic silver, angular symmetry pattern with 1920s decorative-inspired geometry",
    "Gothic rose window reflection pattern in cathedral colors, sacred symmetry design with stained glass-inspired design",
    "Islamic tile reflection design in mosque blue, mathematical symmetry pattern with geometric architecture-inspired design",
    "Chinese paper cut reflection pattern in lucky red, bilateral symmetry design with folk art-inspired design",
    "Japanese origami crane reflection design in pure white, fold symmetry pattern with paper art-inspired geometry",
    "Native American thunderbird reflection pattern in eagle colors, spiritual symmetry design with totemic-inspired design",
    "Egyptian scarab beetle reflection design in pharaonic gold, sacred symmetry pattern with ancient Egyptian-inspired symbolism",
    "Celtic triple spiral reflection pattern in ancient green, rotational symmetry design with druidic-inspired geometry",
    "Hindu yantra reflection design in sacred saffron, divine symmetry pattern with tantric-inspired sacred geometry",
    "Aztec feathered serpent reflection pattern in obsidian black, mythological symmetry design with Quetzalcoatl-inspired design",
    "Persian garden reflection design in royal crimson, symmetry pattern with Middle Eastern garden paradise-inspired design",
    "Russian matryoshka reflection pattern in folk colors, nesting symmetry design with wooden doll-inspired design",
    "African mask reflection design in tribal earth tones, ceremonial symmetry pattern with indigenous-inspired design",
    "Maori tattoo reflection pattern in traditional black, cultural symmetry design with tā moko-inspired design",
    "Viking shield reflection design in warrior colors, heraldic symmetry pattern with Norse-inspired protection design",
    "Venetian carnival mask reflection pattern in festival gold, theatrical symmetry design with masquerade-inspired design",
    "Mexican Talavera reflection design in ceramic blue, pottery symmetry pattern with colonial-inspired artistic design",
    "Portuguese azulejo reflection pattern in tile blue, architectural symmetry design with ceramic-inspired design",
    "Dutch delft reflection design in porcelain white, painted symmetry pattern with ceramic-inspired design",
    "German cuckoo clock reflection pattern in Black Forest green, mechanical symmetry design with timepiece-inspired design",
    "Swiss snowflake reflection design in Alpine white, crystalline symmetry pattern with winter mountain-inspired design",
    "Austrian crystal reflection pattern in transparent clarity, faceted symmetry design with glass-inspired design",
    "Czech Bohemian glass reflection design in rainbow spectrum, optical symmetry pattern with crystal-inspired design",
    "Polish folk paper cut reflection pattern in traditional red, peasant symmetry design with wycinanki-inspired design",
    "Hungarian embroidery reflection design in folk colors, symmetry pattern with traditional Magyar-inspired design",
    "Romanian reflection pattern in Carpathian colors, symmetry design with traditional mountain-inspired design",
    "Bulgarian rose reflection design in Valley of Roses pink, floral symmetry pattern with cultivation-inspired design",
    "Serbian Orthodox cross reflection pattern in Byzantine gold, religious symmetry design with sacred-inspired design",
    "Croatian checkerboard reflection design in national colors, heraldic symmetry pattern with shield-inspired design",
    "Slovenian beehive reflection pattern in honey gold, agricultural symmetry design with apiary-inspired design",
    "Slovakian folk reflection design in mountain colors, traditional symmetry pattern with peasant-inspired design",
    "Lithuanian amber reflection pattern in Baltic gold, geological symmetry design with fossil resin-inspired design",
    "Latvian folk reflection design in forest colors, traditional symmetry pattern with rural Baltic-inspired design",
    "Estonian folk reflection pattern in coastal colors, maritime symmetry design with island-inspired design",
    "Finnish Karelian reflection design in subarctic colors, regional symmetry pattern with forest-inspired design",
    "Norwegian stave church reflection pattern in timber brown, architectural symmetry design with wooden-inspired design",
    "Swedish Dalarna horse reflection design in Falu red, folk symmetry pattern with carved-inspired design",
    "Danish hygge reflection pattern in cozy colors, lifestyle symmetry design with comfort-inspired design",
    "Icelandic saga reflection design in volcanic colors, literary symmetry pattern with epic-inspired design",
    "Faroese knitting reflection pattern in natural colors, symmetry design with fisherman-inspired design",
    "Greenlandic inuksuk reflection design in arctic stone, navigational symmetry pattern with cairn-inspired design",
    "Sami reindeer reflection pattern in aurora colors, nomadic symmetry design with herding-inspired design",
    "Siberian shaman reflection design in tundra colors, spiritual symmetry pattern with vision-inspired design",
    "Mongolian yurt reflection pattern in steppe colors, dwelling symmetry design with nomadic architecture-inspired design"
  ]
}
    
    # Configuration
    config = {
        "output_folder": "/data/generated_carpets",
        "images_per_prompt": 10,  # Total images per prompt (2 seeds × 5 styles = 10)
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