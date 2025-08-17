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

    def generate_pattern_image(
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
        """Generate a single pattern image from prompt."""
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
                "pattern texture, design texture, composition, elements, physical details, "
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
def generate_pattern_dataset():
    """Generate pattern dataset from prompts JSON file."""
    
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
  "Abstract": [
    "Bold splashes of crimson and turquoise creating dynamic non-representational composition, abstract expressionist pattern with irregular color bleeding effects",
    "Sweeping brushstroke-inspired pattern in metallic gold and deep navy, gestural abstract design with spontaneous application technique",
    "Layered translucent color fields in sunset hues, color field style with soft gradients and atmospheric blending",
    "Irregular color blocks with frayed edges in primary colors, hard-edge style with sharp contrasts and bold saturated tones",
    "Dripping paint-inspired pattern in black and white with silver accents, Jackson Pollock inspired splatter technique",
    "Watercolor wash effect in pastel tones with organic bleeds, abstract wet technique creating soft flowing color transitions",
    "Bold diagonal slashes of contrasting colors, Franz Kline inspired black patterns on white ground with minimal color accents",
    "Overlapping circular forms in transparent layers, Kandinsky inspired abstract circles with spiritual color symbolism",
    "Fragmented angular shapes in earthy tones, cubist-inspired abstract design with intersecting planes and faceted surfaces",
    "Fluid pour technique effect in iridescent colors, contemporary abstract design with organic flow patterns and modern art style",
    "Scratched and scraped surfaces revealing underlying colors, graffito technique creating textural abstract composition",
    "Monochromatic tonal variations in charcoal gray, minimalist abstract composition with subtle texture variations and atmospheric depth",
    "Vibrant neon colors in digital glitch-inspired pattern, contemporary cyber abstract composition with pixelated distortions",
    "Gestural marks in earth-pigment colors, primitive abstract design with tribal energy and raw expressive composition",
    "Collaged design fragments in torn organic shapes, mixed media abstract composition with layered textures and found material integration",
    "Spray stencil patterns in street art style, urban abstract design with sharp edges and rebellious graffiti-inspired aesthetic",
    "Marble texture-inspired patterns in natural stone colors, geological abstract composition with veining and mineral formation motifs",
    "Ink blot symmetrical patterns in deep indigo, Rorschach test inspired abstract composition with psychological undertones",
    "Oxidized metal-inspired patterns in rust and verdigris colors, industrial abstract design with patina effects",
    "Kaleidoscope fragment-inspired composition in jewel tones, prismatic abstract composition with crystalline facets and optical illusion effects",
    "Smoke and vapor-inspired patterns in grayscale, ephemeral abstract composition capturing atmospheric movement",
    "Torn paper edge-inspired composition in monochromatic palette, deconstructed abstract composition with negative space emphasis",
    "Paint pouring effect in metallic copper and bronze, liquid metal abstract design with flow patterns and reflective surfaces",
    "Abstract mountain silhouette-inspired composition in purple and orange, landscape-inspired non-representational forms",
    "Gestural calligraphy marks without text meaning, abstract writing composition with expressive line quality",
    "Cracked earth texture-inspired composition in desert colors, natural abstraction with geological patterns",
    "Digital noise-inspired patterns in electric colors, technological abstract composition with data corruption aesthetic",
    "Torn design effect in neutral tones, abstract composition with organic edges and material texture emphasis",
    "Scratched vinyl record-inspired circular patterns in black and gold, music-inspired abstract composition with retro aesthetic",
    "Abstract cityscape impression in industrial colors, urban-inspired non-representational forms",
    "Finger painting technique in primary colors, childlike abstract composition with innocent expression",
    "Abstract lightning bolt patterns in electric blue, energy-inspired abstract composition with dynamic movement",
    "Layered tissue paper effect in translucent pastels, delicate abstract composition with transparency effects",
    "Abstract wave motion in deep ocean blue, water-inspired non-representational movement",
    "Charcoal smudging technique in dramatic gray, expressive abstract composition with tonal gradations",
    "Abstract fire patterns in warm red and orange, flame-inspired energetic composition",
    "Stained glass window-inspired abstract composition in saturated colors, luminous abstract composition with lead line divisions",
    "Abstract cloud formation-inspired patterns in soft white and gray, atmospheric non-representational composition",
    "Paint scraping technique revealing rainbow layers, archaeological abstract composition with stratified color history",
    "Abstract sound wave visualization in electric colors, music-inspired rhythmic patterns",
    "Melted crayon effect in vibrant hues, heat-affected abstract composition with wax flow-inspired patterns",
    "Abstract maze patterns in contrasting colors, labyrinthine non-representational design",
    "Shattered glass refraction-inspired composition in prismatic colors, broken abstract composition with crystalline fragments",
    "Abstract sand dune-inspired patterns in earth tones, desert-inspired flowing forms",
    "Paint bleeding effect in tie dye style, abstract design with organic color migration",
    "Abstract aurora-inspired patterns in cosmic colors, northern lights inspired ethereal composition",
    "Finger-dragged paint effect in rhythmic patterns, gestural abstract composition with tactile quality",
    "Abstract microscopic cell-inspired patterns in scientific colors, biological inspiration composition",
    "Paint roller texture effect in industrial colors, mechanical abstract composition with repetitive mark-making",
    "Abstract earthquake fault line-inspired composition in geological colors, tectonic inspiration with earth movement motifs"
  ],
  "Geometric": [
    "Interlocking hexagonal tessellation in emerald and gold, honeycomb inspired mathematical precision composition",
    "Chevron zigzag pattern in navy and white stripes, arrow-like geometric repetition with dynamic directional movement",
    "Concentric squares rotating at 45-degree angles, nested geometric forms in graduated color progression",
    "Triangular grid pattern in primary colors, equilateral triangle tessellation with Bauhaus design principles",
    "Diamond lattice network in metallic silver and black, rhombus repetition creating optical illusion effects",
    "Greek key meander pattern in classical marble tones, ancient geometric border with continuous interlocking rectangles",
    "Octagonal star patterns in Islamic tile-inspired composition, eight-pointed geometric stars with intricate mathematical precision",
    "Parallel diagonal lines in rainbow gradient colors, linear geometric pattern with color spectrum progression",
    "Pentagon and hexagon combination tessellation, complex geometric puzzle with dual polygon integration",
    "Cube isometric projection pattern in architectural tones, three-dimensional geometric illusion",
    "Circle and square intersection geometry, fundamental shape relationships with overlapping mathematical forms",
    "Triangular spiral formation in fibonacci sequence colors, golden ratio geometric progression",
    "Rectangular grid with alternating color blocks, Mondrian inspired geometric composition",
    "Star polygon patterns in celestial colors, complex geometric stars with multiple pointed formations",
    "Parallel hexagon strips in gradient colors, elongated geometric bands with color transition",
    "Right triangle tessellation in contrasting tones, mathematical puzzle with angular precision composition",
    "Circular sector patterns in pie chart-inspired design, radial geometric divisions with mathematical precision",
    "Square spiral formation in monochromatic scale, geometric progression with mathematical sequence",
    "Diamond grid with circular intersections, hybrid geometric pattern combining angular and curved forms",
    "Triangular wave pattern in oscilloscope-inspired design, geometric sine wave representation",
    "Hexagonal flower of life sacred geometry, ancient geometric symbol with overlapping circles",
    "Rectangular maze pattern in stark black and white, geometric labyrinth with algorithmic path-finding",
    "Pentagon spiral in golden ratio proportions, five-sided geometric progression",
    "Triangular fractal pattern in recursive formation, self-similar geometric repetition composition",
    "Square mandala with rotational symmetry, geometric meditation pattern with four-fold symmetry",
    "Rhombus tessellation in gradient metallic colors, diamond-shaped geometric pattern with three-dimensional shading",
    "Circular grid with square intersections, dual geometric system with curved and angular interaction",
    "Triangular prism optical illusion pattern, three-dimensional geometric perspective",
    "Hexagonal spiral in fibonacci sequence colors, six-sided geometric progression with natural mathematical patterns",
    "Rectangle and circle hybrid tessellation, mixed geometric forms with mathematical precision",
    "Square rotation sequence in time-lapse-inspired design, geometric transformation with mathematical rotation",
    "Diamond checkerboard in high contrast colors, geometric game board pattern with alternating angular forms",
    "Triangular grid with hexagonal gaps, negative space geometric pattern with mathematical precision",
    "Circular sector rainbow in spectrum order, radial geometric color wheel with mathematical color theory",
    "Square fractal border pattern, geometric frame with self-similar mathematical repetition",
    "Pentagon and triangle combination grid, multi-polygon tessellation with complex geometric relationships",
    "Hexagonal honeycomb with gradient fill, natural geometric pattern with mathematical optimization",
    "Rectangular wave interference pattern, geometric wave interaction with mathematical frequency modulation",
    "Triangular kaleidoscope symmetry pattern, geometric reflection with mathematical precision",
    "Diamond grid with triangular subdivision, complex geometric tessellation with mathematical precision",
    "Circular mandala with geometric precision, radial mathematical pattern with perfect symmetrical relationships",
    "Square grid with diagonal intersections, orthogonal geometric system with mathematical precision",
    "Hexagonal prism perspective drawing, three-dimensional geometric projection",
    "Triangular maze with angular pathways, geometric puzzle with mathematical problem-solving complexity",
    "Rectangle and oval intersection pattern, hybrid geometric forms with mathematical precision",
    "Pentagon tessellation with star formation, five-sided geometric pattern with mathematical precision",
    "Circular grid with triangular subdivision, radial geometric pattern with mathematical precision",
    "Square spiral with color progression, geometric sequence with mathematical advancement",
    "Diamond lattice with cubic perspective, three-dimensional geometric pattern with mathematical precision",
    "Triangular grid with hexagonal symmetry, dual geometric system with mathematical harmony"
  ],
  "Damask": [
    "Traditional acanthus leaf scrollwork in rich colors, lustrous design with classical botanical motifs and reversible structure",
    "Baroque scrolling cartouche pattern in gold and burgundy, ornate design with elaborate curved frames and royal decorative tradition",
    "Renaissance palmette and vine design in ivory tones, sophisticated botanical scrollwork with historical craftsmanship",
    "Rococo shell and scroll combination in pearl gray, decorative design with French court elegance and refined asymmetrical balance",
    "Gothic quatrefoil medallion pattern in cathedral colors, medieval design with architectural motifs and ecclesiastical heritage",
    "Byzantine imperial eagle design in royal purple, heraldic pattern with symbolic power and ancient tradition",
    "Art Nouveau flowing tendril pattern in sage green, organic design with sinuous curves and botanical art movement inspiration",
    "Neoclassical urn and garland motif in marble tones, formal design with architectural elements and ancient Greek revival aesthetics",
    "Victorian rose and ribbon combination in dusty pink, elaborate design with sentimental motifs and nineteenth-century romantic sensibility",
    "Elizabethan strapwork pattern in rich burgundy, geometric design with interlaced bands and Tudor period architectural decoration",
    "Louis XIV sun motif in golden yellow, royal French pattern with solar symbolism and absolute monarchy grandeur",
    "Art Deco stepped motif in platinum and black, modernist design with geometric sophistication and machine age luxury",
    "Jacobean crewelwork inspired design in forest green, English country pattern with stylized foliage and rustic elegance",
    "Regency stripe and medallion combination in navy, formal design with neoclassical restraint and British empire sophistication",
    "Federal period eagle and shield design in patriotic colors, American historical pattern with national symbols",
    "Empire style palmette border in imperial red, Napoleonic design with classical motifs and French empire grandeur",
    "Georgian chinoiserie pagoda pattern in blue and white, oriental-inspired design with exotic motifs and colonial trade influence",
    "William Morris inspired design in earth tones, arts and crafts pattern with natural motifs and handcraft revival aesthetics",
    "Edwardian rose garland pattern in cream and gold, delicate design with romantic florals and Edwardian era refinement",
    "Belle Époque serpentine ribbon in champagne, elegant design with flowing curves and French fin de siècle sophistication",
    "Tudor rose and crown combination in royal red, heraldic design with English monarchy symbols and medieval court tradition",
    "Moorish geometric interlace in deep blue, Islamic-inspired design with mathematical precision and Andalusian architectural heritage",
    "Chinese imperial dragon pattern in jade green, oriental design with mythological creatures and dynastic power symbolism",
    "Russian imperial double-headed eagle in gold, czarist design with Byzantine heritage and imperial Russian grandeur",
    "Persian cypress tree motif in jewel tones, middle eastern design with ancient symbols and Islamic garden paradise imagery",
    "Italian Renaissance grotesque pattern in terra cotta, decorative design with fantastical creatures and humanistic art tradition",
    "Spanish colonial fleur-de-lis in silver, ecclesiastical design with religious symbolism and new world missionary aesthetics",
    "German baroque hunting scene in forest colors, narrative design with aristocratic leisure and romantic landscape tradition",
    "Dutch tulip and windmill combination in orange, commercial design with national symbols and golden age prosperity",
    "Portuguese azulejo tile inspired pattern in cobalt, ceramic-inspired design with maritime heritage and exploration age romance",
    "Swedish folk art inspired design in Nordic colors, Scandinavian pattern with rural traditions and democratic craft heritage",
    "Polish sarmatian saber pattern in silver and black, military design with noble warrior tradition and eastern European heritage",
    "Hungarian folk embroidery inspired design in bright red, ethnic pattern with peasant traditions and Carpathian cultural identity",
    "Czech bohemian glass pattern in crystal tones, decorative design with luxury craft tradition and central European sophistication",
    "Austrian alpine edelweiss motif in mountain colors, regional design with natural symbols and Habsburg empire romanticism",
    "Swiss clockwork gear pattern in precision gray, mechanical design with craft tradition and alpine engineering excellence",
    "Belgian lace inspired design in ivory white, pattern with handcraft excellence and Flemish artistic tradition",
    "Scottish tartan inspired composition in clan colors, geometric design with highland tradition and Celtic cultural identity",
    "Irish Celtic knotwork pattern in emerald, interlaced design with ancient symbols and Gaelic artistic heritage",
    "Welsh dragon and leek combination in national colors, heraldic design with mythological symbols and Celtic tradition",
    "Cornish tin mine inspired pattern in metallic tones, industrial design with mining heritage and coastal Celtic culture",
    "Manx triskelion spiral in Isle of Man colors, ancient design with Celtic symbols and Viking cultural fusion",
    "Breton sailor stripe adaptation in navy and white, maritime design with coastal tradition and French provincial heritage",
    "Provençal lavender field pattern in purple tones, regional design with agricultural heritage and Mediterranean luxury tradition",
    "Tuscan vineyard inspired design in wine colors, Italian pattern with agricultural romance and Renaissance cultural refinement",
    "Andalusian olive grove motif in golden green, Spanish design with agricultural tradition and Moorish cultural synthesis",
    "Venetian carnival mask pattern in jewel tones, theatrical design with artistic celebration and Italian renaissance pageantry",
    "Florentine lily and shield combination in red and gold, heraldic design with republican tradition and Renaissance artistic excellence",
    "Milanese fashion inspired pattern in luxury colors, contemporary design with style tradition and Italian design sophistication",
    "Neapolitan volcano inspired design in lava tones, geological pattern with natural drama and southern Italian passionate temperament"
  ],
  "Floral": [
    "English cottage garden roses in delicate composition, climbing rose motifs with soft pink and cream tones in traditional pattern",
    "Japanese cherry blossom branches in minimalist style, delicate sakura petals with zen-inspired composition",
    "French Provençal lavender fields, purple flower spikes with Mediterranean countryside charm",
    "Dutch tulip garden in botanical composition, colorful tulip varieties with detailed scientific accuracy",
    "Indian lotus pond in traditional style, sacred lotuses with gold accents and spiritual symbolism",
    "Chinese peony garden composition, luxurious peonies with flowing brushstroke-inspired patterns and imperial elegance",
    "Persian flower medallions in jewel tones, stylized blooms with intricate detail and cultural heritage",
    "Art Nouveau poppy field composition, flowing poppies with organic curves and decorative art movement aesthetics",
    "Victorian bouquet in romantic style, mixed flowers with ribbon and lace-inspired details in sentimental arrangement",
    "Tropical hibiscus paradise in vivid colors, exotic blooms with bold petal patterns and island paradise atmosphere",
    "Wild meadow flowers in folk art composition, naive flowering field with charming simplicity and rustic appeal",
    "Moroccan orange blossom pattern in geometric style, stylized citrus flowers with Islamic art influence",
    "Scottish heather moorland composition, purple heather blooms with misty highland atmosphere",
    "Spanish flamenco rose in passionate red, dramatic single bloom motif with cultural dance inspiration",
    "Tuscan sunflower field, golden sunflowers with van Gogh inspired patterns in warm tones",
    "Brazilian orchid greenhouse in tropical composition, exotic orchid varieties with lush jungle atmosphere",
    "Russian folk flower painting in bright colors, traditional decorative blooms with Slavic cultural patterns",
    "Australian wildflower bush in botanical style, native flowering plants with outback natural beauty in earth tones",
    "Mexican marigold celebration in festival colors, vibrant marigolds with Day of the Dead cultural significance",
    "Nordic summer flowers in minimal style, simple blooms with Scandinavian design restraint and natural purity",
    "Egyptian papyrus flower pattern in ancient style, stylized river blooms with hieroglyphic artistic tradition",
    "Greek island bougainvillea in Mediterranean style, climbing flowering vines with coastal whitewash architecture inspiration",
    "Canadian maple blossom in seasonal composition, delicate tree flowers with autumn color anticipation",
    "Korean magnolia garden in traditional style, elegant magnolia blooms with Asian garden design principles",
    "German alpine flower meadow in realistic style, mountain wildflowers with precise botanical illustration",
    "Irish shamrock field in symbolic composition, three-leaf clovers with cultural identity and luck symbolism in green",
    "Italian Renaissance garden in formal style, geometric flower beds with classical garden design principles",
    "Turkish tulip pattern in traditional style, stylized tulips with Ottoman empire artistic heritage",
    "Portuguese azalea garden in coastal style, blooming azaleas with Atlantic maritime garden influence",
    "Argentinian jacaranda tree in dramatic style, purple flowering tree with South American tropical grandeur",
    "New Zealand pohutukawa in native style, red flowering tree with Maori cultural significance and coastal beauty",
    "South African protea in bold style, exotic protea blooms with unique African botanical character",
    "Thai orchid temple garden in spiritual style, sacred orchids with Buddhist temple garden serenity",
    "Lebanese cedar flower in mountain style, high altitude blooms with Middle Eastern alpine character",
    "Filipino sampaguita garland in cultural style, national flower chains with tropical island tradition",
    "Venezuelan bird of paradise in exotic style, dramatic tropical blooms with South American rainforest luxury",
    "Cambodian lotus temple pond in sacred style, religious lotus blooms with Angkor temple spirituality",
    "Malaysian hibiscus national flower in tropical style, state flower with Southeast Asian cultural pride",
    "Indonesian frangipani in temple style, sacred temple flowers with Hindu-Buddhist spiritual significance",
    "Bangladeshi water lily in monsoon style, floating blooms with seasonal flooding natural adaptation",
    "Sri Lankan blue lotus in ancient style, sacred blooms with Buddhist temple garden meditation atmosphere",
    "Nepalese rhododendron mountain in Himalayan style, high altitude blooms with mountain kingdom natural beauty",
    "Bhutanese blue poppy in rare style, national flower with Himalayan kingdom unique botanical treasure",
    "Maldivian pink rose in island style, tropical adapted roses with coral island paradise romance",
    "Seychelles coco de mer palm flower in exotic style, rare island blooms with oceanic isolation uniqueness",
    "Madagascar periwinkle in endemic style, island evolution blooms with unique Malagasy botanical heritage",
    "Mauritius trochetia in national style, endemic island flower with volcanic island botanical adaptation",
    "Fiji bougainvillea in Pacific style, tropical climbing blooms with South Pacific island paradise beauty",
    "Samoa tiare flower in Polynesian style, traditional island blooms with Pacific cultural lei-making tradition",
    "Tonga heilala in royal style, kingdom national flower with Polynesian royal garden ceremonial significance"
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
            "pattern texture, design texture, composition, physical material, "
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
    
    print(f"Starting pattern generation for {len(prompts_data)} classes")
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
                        image, actual_seed = generator.generate_pattern_image(
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
        generate_pattern_dataset.remote()