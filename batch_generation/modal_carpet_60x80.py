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

app = modal.App("carpet-60x80-generator", image=image)

# Model configuration
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16

# Volume for storing generated images - 60x80 specific
output_volume = modal.Volume.from_name("generated-carpet-60x80", create_if_missing=True)

# Target and generation resolutions
TARGET_WIDTH = 60
TARGET_HEIGHT = 80
GEN_WIDTH = 64  # 16-divisible
GEN_HEIGHT = 80  # 16-divisible

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
def generate_carpet_dataset_60x80():
    """Generate carpet dataset from prompts JSON file."""
    
    # Define artistic styles to prepend to prompts
    artistic_styles = [
        "Watercolor: soft translucent color washes with organic bleeds, gradual tone transitions, and spontaneous edge effects that create flowing, unpredictable boundaries",
        "Oil Paint: rich lustrous colors with thick impasto-like depth, visible directional brushstroke patterns, and layered color buildups that create dimensional surface quality",
        "Block Print: hand-carved aesthetic with bold graphic silhouettes, intentional registration shifts, carved texture marks, and the authentic irregularities of relief printing",
        "Ink Drawing: bold black linework with varied line weights, crosshatching density variations, stippling dot patterns, and high contrast between positive and negative space",
        "Digital Vector: mathematically precise curves with perfect color fills, scalable geometric precision, gradient meshes, and flawless symmetrical repetitions",
        "Batik: wax-resist crackle effects with organic dye penetration, flowing color boundaries, traditional resist patterns, and the characteristic veined appearance of cracked wax",
        "Screen Print: flat uniform color fields with razor-sharp edges, perfect opacity, halftone dot patterns for tonal variation, and the graphic boldness of commercial printing",
        "Acrylic: vibrant matte colors with quick-drying crisp edges, layered opacity, and bold color saturation with clean graphic precision",
        "Pastel: soft powdery pigments with blended atmospheric effects, velvety texture buildup, and delicate color transitions that create luminous depth",
        "Pointillism: pure color dots that create optical color mixing, textural surface interest, and luminous effects through careful dot placement and spacing"
        ]
    
    # Test prompts optimized for FLUX seamless pattern generation
    prompts_data = {
  "Abstract": [
    "Bold splashes of crimson and turquoise dye creating dynamic non-representational textile composition, abstract expressionist weave pattern with irregular color bleeding effects",
    "Sweeping brushstroke-inspired woven pattern in metallic gold and deep navy threads, gestural abstract textile with spontaneous dye application technique",
    "Layered translucent color fields in sunset hues woven into fabric, color field textile style with soft gradients and atmospheric thread blending",
    "Irregular color blocks with frayed edges in primary colors, hard-edge weaving style with sharp contrasts and bold saturated yarn tones",
    "Dripping paint-inspired textile pattern in black and white with silver thread accents, Jackson Pollock inspired splatter weave technique",
    "Watercolor wash effect in pastel yarn tones with organic bleeds, abstract wet-dye technique creating soft flowing color transitions in fabric",
    "Bold diagonal slashes of contrasting woven colors, Franz Kline inspired black thread patterns on white ground with minimal color accents",
    "Overlapping circular forms in transparent thread layers, Kandinsky inspired abstract circles woven with spiritual color symbolism",
    "Fragmented angular shapes in earthy yarn tones, cubist-inspired abstract weave with intersecting planes and faceted textile surfaces",
    "Fluid pour technique effect in iridescent threads, contemporary abstract weaving with organic flow patterns and modern textile art style",
    "Scratched and scraped dye surfaces revealing underlying thread colors, textile graffito technique creating textural abstract weave composition",
    "Monochromatic tonal variations in charcoal gray threads, minimalist abstract textile with subtle texture variations and atmospheric depth",
    "Vibrant neon yarn colors in digital glitch-inspired pattern, contemporary cyber abstract weave with pixelated distortions",
    "Hand-painted gestural marks translated to earth-pigment dyed fabric, primitive abstract textile with tribal energy and raw expressive weaving",
    "Collaged fabric fragments in torn organic shapes, mixed media abstract textile with layered textures and found material integration",
    "Spray-dyed stencil patterns in street art style, urban abstract textile with sharp edges and rebellious graffiti-inspired aesthetic",
    "Marble texture-inspired woven patterns in natural stone colors, geological abstract textile with veining and mineral formation motifs",
    "Ink blot symmetrical patterns in deep indigo yarn, Rorschach test inspired abstract weave with psychological undertones",
    "Oxidized metal-inspired textile patterns in rust and verdigris dyes, industrial abstract weaving with patina effects",
    "Kaleidoscope fragment-inspired weave in jewel tones, prismatic abstract textile with crystalline facets and optical illusion effects",
    "Smoke and vapor-inspired patterns in grayscale threads, ephemeral abstract textile capturing atmospheric movement in fabric",
    "Torn paper edge-inspired weave in monochromatic palette, deconstructed abstract textile with negative space emphasis",
    "Paint pouring effect in metallic copper and bronze threads, liquid metal abstract weaving with flow patterns and reflective surfaces",
    "Abstract mountain silhouette-inspired weave in purple and orange, landscape-inspired non-representational textile forms",
    "Gestural calligraphy marks without text meaning woven into fabric, abstract writing textile with expressive line quality",
    "Cracked earth texture-inspired weave in desert colors, natural abstraction textile with geological patterns",
    "Digital noise-inspired patterns in electric colored yarns, technological abstract weave with data corruption aesthetic",
    "Hand-torn fabric effect in neutral tones, textile abstract with organic edges and material texture emphasis",
    "Scratched vinyl record-inspired circular patterns in black and gold thread, music-inspired abstract weave with retro aesthetic",
    "Abstract cityscape impression in industrial colored yarns, urban-inspired non-representational textile forms",
    "Finger painting technique translated to primary colored threads, childlike abstract textile with innocent expression",
    "Abstract lightning bolt patterns in electric blue yarn, energy-inspired abstract weave with dynamic movement",
    "Layered tissue paper effect in translucent thread pastels, delicate abstract textile with transparency effects",
    "Abstract wave motion in deep ocean blue threads, water-inspired non-representational movement woven into fabric",
    "Charcoal smudging technique in dramatic gray yarns, expressive abstract textile with tonal gradations",
    "Abstract fire patterns in warm red and orange threads, flame-inspired energetic textile composition",
    "Stained glass window-inspired abstract weave in saturated colors, luminous abstract textile with lead line divisions",
    "Abstract cloud formation-inspired patterns in soft white and gray threads, atmospheric non-representational textile",
    "Paint scraping technique revealing rainbow thread layers, archaeological abstract weave with stratified color history",
    "Abstract sound wave visualization in electric colored threads, music-inspired rhythmic textile patterns",
    "Melted crayon effect in vibrant yarn hues, heat-affected abstract weave with wax flow-inspired patterns",
    "Abstract maze patterns in contrasting colored threads, labyrinthine non-representational textile design",
    "Shattered glass refraction-inspired weave in prismatic colors, broken abstract textile with crystalline fragments",
    "Abstract sand dune-inspired patterns in earth-toned threads, desert-inspired flowing textile forms",
    "Paint bleeding effect through fabric in tie-dye style, textile abstract with organic color migration",
    "Abstract aurora-inspired patterns in cosmic colored threads, northern lights inspired ethereal textile composition",
    "Finger-dragged paint effect in rhythmic thread patterns, gestural abstract textile with tactile quality",
    "Abstract microscopic cell-inspired patterns in scientific colored yarns, biological inspiration woven textile",
    "Paint roller texture effect in industrial colored threads, mechanical abstract weave with repetitive mark-making",
    "Abstract earthquake fault line-inspired weave in geological colors, tectonic inspiration textile with earth movement motifs"
  ],
  "Geometric": [
    "Interlocking hexagonal tessellation woven in emerald and gold threads, honeycomb inspired mathematical precision textile",
    "Chevron zigzag pattern in navy and white woven stripes, arrow-like geometric repetition with dynamic directional movement",
    "Concentric squares rotating at 45-degree angles in woven threads, nested geometric forms in graduated color progression",
    "Triangular grid pattern in primary colored yarns, equilateral triangle tessellation with Bauhaus design principles",
    "Diamond lattice network in metallic silver and black threads, rhombus repetition creating optical illusion textile effects",
    "Greek key meander pattern woven in classical marble-toned threads, ancient geometric border with continuous interlocking rectangles",
    "Octagonal star patterns in Islamic tile-inspired weave, eight-pointed geometric stars with intricate mathematical precision",
    "Parallel diagonal lines in rainbow gradient threads, linear geometric pattern with color spectrum progression",
    "Pentagon and hexagon combination tessellation in woven fabric, complex geometric puzzle with dual polygon integration",
    "Cube isometric projection pattern in architectural-toned threads, three-dimensional geometric illusion textile",
    "Circle and square intersection geometry in woven fabric, fundamental shape relationships with overlapping mathematical forms",
    "Triangular spiral formation in fibonacci sequence threads, golden ratio geometric progression textile",
    "Rectangular grid with alternating color thread blocks, Mondrian inspired geometric composition weave",
    "Star polygon patterns in celestial colored threads, complex geometric stars with multiple pointed formations",
    "Parallel hexagon strips in gradient colored yarns, elongated geometric bands with color transition",
    "Right triangle tessellation in contrasting thread tones, mathematical puzzle with angular precision weave",
    "Circular sector patterns in pie chart-inspired threads, radial geometric divisions with mathematical precision",
    "Square spiral formation in monochromatic thread scale, geometric progression with mathematical sequence",
    "Diamond grid with circular intersections in woven fabric, hybrid geometric pattern combining angular and curved forms",
    "Triangular wave pattern in oscilloscope-inspired threads, geometric sine wave representation textile",
    "Hexagonal flower of life sacred geometry in woven threads, ancient geometric symbol with overlapping circles",
    "Rectangular maze pattern in stark black and white threads, geometric labyrinth with algorithmic path-finding",
    "Pentagon spiral in golden ratio proportioned threads, five-sided geometric progression textile",
    "Triangular fractal pattern in recursive formation threads, self-similar geometric repetition weave",
    "Square mandala with rotational symmetry in woven threads, geometric meditation pattern with four-fold symmetry",
    "Rhombus tessellation in gradient metallic threads, diamond-shaped geometric pattern with three-dimensional shading",
    "Circular grid with square intersections in woven fabric, dual geometric system with curved and angular interaction",
    "Triangular prism optical illusion pattern in threads, three-dimensional geometric perspective textile",
    "Hexagonal spiral in fibonacci sequence threads, six-sided geometric progression with natural mathematical patterns",
    "Rectangle and circle hybrid tessellation in woven fabric, mixed geometric forms with mathematical precision",
    "Square rotation sequence in time-lapse-inspired threads, geometric transformation with mathematical rotation",
    "Diamond checkerboard in high contrast threads, geometric game board pattern with alternating angular forms",
    "Triangular grid with hexagonal gaps in woven fabric, negative space geometric pattern with mathematical precision",
    "Circular sector rainbow in spectrum order threads, radial geometric color wheel with mathematical color theory",
    "Square fractal border pattern in threads, geometric frame with self-similar mathematical repetition",
    "Pentagon and triangle combination grid in woven fabric, multi-polygon tessellation with complex geometric relationships",
    "Hexagonal honeycomb with gradient fill threads, natural geometric pattern with mathematical optimization",
    "Rectangular wave interference pattern in threads, geometric wave interaction with mathematical frequency modulation",
    "Triangular kaleidoscope symmetry pattern in woven fabric, geometric reflection with mathematical precision",
    "Diamond grid with triangular subdivision in threads, complex geometric tessellation with mathematical precision",
    "Circular mandala with geometric precision in woven threads, radial mathematical pattern with perfect symmetrical relationships",
    "Square grid with diagonal intersections in fabric, orthogonal geometric system with mathematical precision",
    "Hexagonal prism perspective drawing in threads, three-dimensional geometric projection textile",
    "Triangular maze with angular pathways in woven fabric, geometric puzzle with mathematical problem-solving complexity",
    "Rectangle and oval intersection pattern in threads, hybrid geometric forms with mathematical precision",
    "Pentagon tessellation with star formation in woven fabric, five-sided geometric pattern with mathematical precision",
    "Circular grid with triangular subdivision in threads, radial geometric pattern with mathematical precision",
    "Square spiral with color progression in woven fabric, geometric sequence with mathematical advancement",
    "Diamond lattice with cubic perspective in threads, three-dimensional geometric pattern with mathematical precision",
    "Triangular grid with hexagonal symmetry in woven fabric, dual geometric system with mathematical harmony"
  ],
  "Damask": [
    "Traditional acanthus leaf scrollwork in silk damask weave technique, lustrous fabric with classical botanical motifs and reversible structure",
    "Baroque scrolling cartouche pattern in gold and burgundy damask, ornate textile with elaborate curved frames and royal decorative tradition",
    "Renaissance palmette and vine damask in ivory silk tones, sophisticated botanical scrollwork with historical weaving craftsmanship",
    "Rococo shell and scroll combination in pearl gray damask, decorative textile with French court elegance and refined asymmetrical balance",
    "Gothic quatrefoil medallion pattern in cathedral-colored damask, medieval textile with architectural motifs and ecclesiastical design heritage",
    "Byzantine imperial eagle damask in royal purple silk, heraldic textile pattern with symbolic power and ancient weaving tradition",
    "Art Nouveau flowing tendril pattern in sage green damask, organic textile with sinuous curves and botanical art movement inspiration",
    "Neoclassical urn and garland motif in marble-toned damask, formal textile with architectural elements and ancient Greek revival aesthetics",
    "Victorian rose and ribbon combination in dusty pink damask, elaborate textile with sentimental motifs and nineteenth-century romantic sensibility",
    "Elizabethan strapwork pattern in rich burgundy damask, geometric textile with interlaced bands and Tudor period architectural decoration",
    "Louis XIV sun motif damask in golden yellow silk, royal French textile pattern with solar symbolism and absolute monarchy grandeur",
    "Art Deco stepped motif in platinum and black damask, modernist textile with geometric sophistication and machine age luxury",
    "Jacobean crewelwork inspired damask in forest green, English country textile pattern with stylized foliage and rustic elegance",
    "Regency stripe and medallion combination in navy damask, formal textile with neoclassical restraint and British empire sophistication",
    "Federal period eagle and shield damask in patriotic colors, American historical textile pattern with national symbols",
    "Empire style palmette border in imperial red damask, Napoleonic textile with classical motifs and French empire grandeur",
    "Georgian chinoiserie pagoda pattern in blue and white damask, oriental-inspired textile with exotic motifs and colonial trade influence",
    "William Morris inspired damask in earth tones, arts and crafts textile pattern with natural motifs and handcraft revival aesthetics",
    "Edwardian rose garland pattern in cream and gold damask, delicate textile with romantic florals and Edwardian era refinement",
    "Belle Époque serpentine ribbon in champagne damask, elegant textile with flowing curves and French fin de siècle sophistication",
    "Tudor rose and crown combination in royal red damask, heraldic textile with English monarchy symbols and medieval court tradition",
    "Moorish geometric interlace in deep blue damask, Islamic-inspired textile with mathematical precision and Andalusian architectural heritage",
    "Chinese imperial dragon pattern in jade green damask, oriental textile with mythological creatures and dynastic power symbolism",
    "Russian imperial double-headed eagle in gold damask, czarist textile with Byzantine heritage and imperial Russian grandeur",
    "Persian cypress tree motif in jewel-toned damask, middle eastern textile with ancient symbols and Islamic garden paradise imagery",
    "Italian Renaissance grotesque pattern in terra cotta damask, decorative textile with fantastical creatures and humanistic art tradition",
    "Spanish colonial fleur-de-lis in silver damask, ecclesiastical textile with religious symbolism and new world missionary aesthetics",
    "German baroque hunting scene in forest-colored damask, narrative textile with aristocratic leisure and romantic landscape tradition",
    "Dutch tulip and windmill combination in orange damask, commercial textile with national symbols and golden age prosperity",
    "Portuguese azulejo tile inspired pattern in cobalt damask, ceramic-inspired textile with maritime heritage and exploration age romance",
    "Swedish folk art inspired damask in Nordic colors, Scandinavian textile pattern with rural traditions and democratic craft heritage",
    "Polish sarmatian saber pattern in silver and black damask, military textile with noble warrior tradition and eastern European heritage",
    "Hungarian folk embroidery inspired damask in bright red, ethnic textile pattern with peasant traditions and Carpathian cultural identity",
    "Czech bohemian glass pattern in crystal-toned damask, decorative textile with luxury craft tradition and central European sophistication",
    "Austrian alpine edelweiss motif in mountain-colored damask, regional textile with natural symbols and Habsburg empire romanticism",
    "Swiss clockwork gear pattern in precision gray damask, mechanical textile with craft tradition and alpine engineering excellence",
    "Belgian lace inspired damask in ivory white, textile pattern with handcraft excellence and Flemish artistic tradition",
    "Scottish tartan inspired weave in clan-colored damask, geometric textile with highland tradition and Celtic cultural identity",
    "Irish Celtic knotwork pattern in emerald damask, interlaced textile with ancient symbols and Gaelic artistic heritage",
    "Welsh dragon and leek combination in national-colored damask, heraldic textile with mythological symbols and Celtic tradition",
    "Cornish tin mine inspired pattern in metallic-toned damask, industrial textile with mining heritage and coastal Celtic culture",
    "Manx triskelion spiral in Isle of Man colored damask, ancient textile with Celtic symbols and Viking cultural fusion",
    "Breton sailor stripe adaptation in navy and white damask, maritime textile with coastal tradition and French provincial heritage",
    "Provençal lavender field pattern in purple-toned damask, regional textile with agricultural heritage and Mediterranean luxury tradition",
    "Tuscan vineyard inspired damask in wine colors, Italian textile pattern with agricultural romance and Renaissance cultural refinement",
    "Andalusian olive grove motif in golden green damask, Spanish textile with agricultural tradition and Moorish cultural synthesis",
    "Venetian carnival mask pattern in jewel-toned damask, theatrical textile with artistic celebration and Italian renaissance pageantry",
    "Florentine lily and shield combination in red and gold damask, heraldic textile with republican tradition and Renaissance artistic excellence",
    "Milanese fashion inspired pattern in luxury-colored damask, contemporary textile with style tradition and Italian design sophistication",
    "Neapolitan volcano inspired damask in lava-toned colors, geological textile pattern with natural drama and southern Italian passionate temperament"
  ],
  "Floral": [
    "English cottage garden roses in delicate rug pile, climbing rose motifs with soft pink and cream woolen tones in traditional carpet weave",
    "Japanese cherry blossom branches in minimalist tapestry style, delicate sakura petals woven with zen-inspired composition in silk threads",
    "French Provençal lavender fields in hand-knotted rug, purple flower spikes woven with Mediterranean countryside charm in woolen pile",
    "Dutch tulip garden in botanical tapestry weave, colorful tulip varieties hand-woven with detailed scientific accuracy in fine threads",
    "Indian lotus pond in traditional carpet style, sacred lotuses woven with gold thread accents and spiritual symbolism in silk pile",
    "Chinese peony garden in silk carpet weave, luxurious peonies hand-knotted with flowing brushstroke-inspired patterns and imperial elegance",
    "Persian carpet flower medallions in jewel-toned wool, stylized blooms hand-woven with intricate detail and cultural heritage in traditional knots",
    "Art Nouveau poppy field in tapestry weave, flowing poppies woven with organic curves and decorative art movement aesthetics in fine threads",
    "Victorian bouquet in romantic carpet style, mixed flowers hand-knotted with ribbon and lace-inspired details in sentimental arrangement",
    "Tropical hibiscus paradise in vivid woolen rug, exotic blooms woven with bold petal patterns and island paradise atmosphere in dense pile",
    "Wild meadow flowers in folk art carpet weave, naive flowering field hand-knotted with charming simplicity and rustic appeal in country colors",
    "Moroccan orange blossom pattern in geometric rug style, stylized citrus flowers woven with Islamic art influence in traditional carpet knots",
    "Scottish heather moorland in wool carpet weave, purple heather blooms hand-knotted with misty highland atmosphere in thick pile",
    "Spanish flamenco rose in passionate red carpet, dramatic single bloom motif woven with cultural dance inspiration in silk and wool",
    "Tuscan sunflower field in hand-knotted rug style, golden sunflowers woven with van Gogh inspired patterns in warm woolen threads",
    "Brazilian orchid greenhouse in tropical carpet weave, exotic orchid varieties hand-knotted with lush jungle atmosphere in silk pile",
    "Russian folk flower painting in bright carpet colors, traditional decorative blooms woven with Slavic cultural patterns in woolen threads",
    "Australian wildflower bush in botanical carpet style, native flowering plants hand-knotted with outback natural beauty in earth-toned wool",
    "Mexican marigold celebration in festival carpet colors, vibrant marigolds woven with Day of the Dead cultural significance in bright threads",
    "Nordic summer flowers in minimal carpet style, simple blooms hand-knotted with Scandinavian design restraint and natural purity in wool",
    "Egyptian papyrus flower pattern in ancient carpet style, stylized river blooms woven with hieroglyphic artistic tradition in linen threads",
    "Greek island bougainvillea in Mediterranean carpet style, climbing flowering vines hand-knotted with coastal whitewash architecture inspiration",
    "Canadian maple blossom in seasonal carpet weave, delicate tree flowers woven with autumn color anticipation in natural wool",
    "Korean magnolia garden in traditional carpet style, elegant magnolia blooms hand-knotted with Asian garden design principles in silk",
    "German alpine flower meadow in realistic carpet style, mountain wildflowers woven with precise botanical illustration in fine wool",
    "Irish shamrock field in symbolic carpet weave, three-leaf clovers hand-knotted with cultural identity and luck symbolism in green wool",
    "Italian Renaissance garden in formal carpet style, geometric flower beds woven with classical garden design principles in silk threads",
    "Turkish carpet tulip pattern in traditional style, stylized tulips hand-knotted with Ottoman empire artistic heritage in classic knots",
    "Portuguese azalea garden in coastal carpet style, blooming azaleas woven with Atlantic maritime garden influence in wool pile",
    "Argentinian jacaranda tree in dramatic carpet style, purple flowering tree hand-knotted with South American tropical grandeur in silk",
    "New Zealand pohutukawa in native carpet style, red flowering tree woven with Maori cultural significance and coastal beauty",
    "South African protea in bold carpet style, exotic protea blooms hand-knotted with unique African botanical character in wool",
    "Thai orchid temple garden in spiritual carpet style, sacred orchids woven with Buddhist temple garden serenity in silk threads",
    "Lebanese cedar flower in mountain carpet style, high altitude blooms hand-knotted with Middle Eastern alpine character in wool",
    "Filipino sampaguita garland in cultural carpet style, national flower chains woven with tropical island tradition in fine threads",
    "Venezuelan bird of paradise in exotic carpet style, dramatic tropical blooms hand-knotted with South American rainforest luxury",
    "Cambodian lotus temple pond in sacred carpet style, religious lotus blooms woven with Angkor temple spirituality in silk",
    "Malaysian hibiscus national flower in tropical carpet style, state flower hand-knotted with Southeast Asian cultural pride in wool",
    "Indonesian frangipani in temple carpet style, sacred temple flowers woven with Hindu-Buddhist spiritual significance in silk threads",
    "Bangladeshi water lily in monsoon carpet style, floating blooms hand-knotted with seasonal flooding natural adaptation in cotton",
    "Sri Lankan blue lotus in ancient carpet style, sacred blooms woven with Buddhist temple garden meditation atmosphere in silk",
    "Nepalese rhododendron mountain in Himalayan carpet style, high altitude blooms hand-knotted with mountain kingdom natural beauty",
    "Bhutanese blue poppy in rare carpet style, national flower woven with Himalayan kingdom unique botanical treasure in wool",
    "Maldivian pink rose in island carpet style, tropical adapted roses hand-knotted with coral island paradise romance in silk",
    "Seychelles coco de mer palm flower in exotic carpet style, rare island blooms woven with oceanic isolation uniqueness",
    "Madagascar periwinkle in endemic carpet style, island evolution blooms hand-knotted with unique Malagasy botanical heritage",
    "Mauritius trochetia in national carpet style, endemic island flower woven with volcanic island botanical adaptation in wool",
    "Fiji bougainvillea in Pacific carpet style, tropical climbing blooms hand-knotted with South Pacific island paradise beauty",
    "Samoa tiare flower in Polynesian carpet style, traditional island blooms woven with Pacific cultural lei-making tradition",
    "Tonga heilala in royal carpet style, kingdom national flower hand-knotted with Polynesian royal garden ceremonial significance"
  ],
  "Border": [
    "Greek key meander border in classical marble white thread, continuous geometric pattern woven with ancient architectural heritage",
    "Celtic knotwork border in emerald green yarn, interlaced patterns hand-woven with Irish cultural heritage and endless spiritual symbolism",
    "Art Deco sunburst border in gold and black thread, radiating patterns woven with 1920s luxury aesthetics and machine age glamour",
    "Byzantine mosaic border in jewel-toned threads, tessellated patterns hand-woven with Eastern Orthodox religious art and imperial grandeur",
    "Egyptian hieroglyph border in sandstone-colored yarn, ancient symbols woven with pharaonic heritage and Nile valley civilization",
    "Chinese cloud scroll border in imperial yellow thread, flowing patterns hand-woven with Mandate of Heaven symbolism and dynastic art",
    "Islamic geometric star border in turquoise yarn, mathematical patterns woven with Middle Eastern architectural heritage and spiritual geometry",
    "Art Nouveau vine border in organic green thread, flowing botanical patterns hand-woven with turn-of-century artistic movement aesthetics",
    "Persian palmette border in royal blue yarn, stylized leaf patterns woven with ancient Middle Eastern textile tradition and garden paradise",
    "Roman acanthus leaf border in terra cotta thread, classical botanical patterns hand-woven with empire architectural decoration",
    "Japanese wave border in indigo blue yarn, stylized water patterns woven with ukiyo-e artistic tradition and oceanic spirituality",
    "Indian paisley border in saffron orange thread, teardrop patterns hand-woven with Mughal empire heritage and Kashmir textile tradition",
    "Gothic trefoil border in cathedral gray yarn, three-leaf patterns woven with medieval architecture and ecclesiastical symbolism",
    "Russian folk flower border in bright red thread, traditional botanical patterns hand-woven with Slavic cultural heritage and peasant art",
    "Moorish tile border in azure blue yarn, geometric patterns woven with Andalusian architectural heritage and Islamic Spain legacy",
    "Viking rune border in iron gray thread, ancient symbols hand-woven with Norse cultural heritage and Scandinavian warrior tradition",
    "Aztec step-fret border in obsidian black yarn, geometric patterns woven with Mesoamerican architectural heritage and solar symbolism",
    "Turkish tulip border in Ottoman red thread, stylized flower patterns hand-woven with imperial garden tradition and Anatolian textile heritage",
    "French fleur-de-lis border in royal purple yarn, heraldic patterns woven with Bourbon monarchy symbolism and courtly elegance",
    "Scottish tartan border in clan-colored threads, plaid patterns hand-woven with Highland heritage and Gaelic cultural identity",
    "Welsh dragon border in national red yarn, mythological patterns woven with Celtic heritage and Cymric cultural pride",
    "Portuguese azulejo border in cobalt blue thread, ceramic tile-inspired patterns hand-woven with maritime heritage and exploration romance",
    "Spanish Mudéjar border in burnished gold yarn, hybrid patterns woven with Christian-Islamic synthesis and Iberian cultural fusion",
    "Dutch delft border in ceramic blue thread, painted pottery-inspired patterns hand-woven with Golden Age prosperity and maritime heritage",
    "German Gothic border in cathedral stone-colored yarn, architectural patterns woven with Holy Roman Empire heritage and medieval craftsmanship",
    "Italian Renaissance border in Medici red thread, classical patterns hand-woven with humanist revival and Florentine artistic excellence",
    "English Tudor border in royal green yarn, heraldic patterns woven with monarchy heritage and Anglican cultural tradition",
    "Polish Sarmatian border in noble silver thread, aristocratic patterns hand-woven with Commonwealth heritage and eastern European grandeur",
    "Hungarian folk border in paprika red yarn, traditional patterns woven with Magyar heritage and Carpathian cultural identity",
    "Czech Bohemian border in crystal clear thread, decorative patterns hand-woven with glass-making tradition and central European sophistication",
    "Austrian Alpine border in mountain white yarn, regional patterns woven with Habsburg empire heritage and Alpine cultural tradition",
    "Swiss mountain border in snow white thread, geometric patterns hand-woven with confederation heritage and Alpine precision craftsmanship",
    "Belgian lace border in ivory cream yarn, delicate patterns woven with Flemish heritage and textile artistry excellence",
    "Scandinavian rune border in fjord blue thread, ancient patterns hand-woven with Viking heritage and Nordic cultural minimalism",
    "Latvian folk border in amber yellow yarn, traditional patterns woven with Baltic heritage and forest culture symbolism",
    "Lithuanian cross border in medieval gold thread, sacred patterns hand-woven with Catholic heritage and Baltic cultural resistance",
    "Estonian folk border in Baltic blue yarn, coastal patterns woven with Finno-Ugric heritage and maritime cultural tradition",
    "Finnish Karelian border in forest green thread, regional patterns hand-woven with subarctic heritage and Finno-Ugric cultural identity",
    "Norwegian stave border in timber brown yarn, architectural patterns woven with wooden church heritage and fjord cultural tradition",
    "Swedish folk border in Falu red thread, traditional patterns hand-woven with Nordic heritage and Scandinavian cultural simplicity",
    "Danish Viking border in North Sea gray yarn, ancient patterns woven with seafaring heritage and Nordic cultural minimalism",
    "Icelandic saga border in volcanic black thread, literary-inspired patterns hand-woven with island heritage and Nordic cultural isolation",
    "Faroese fisherman border in oceanic blue yarn, maritime patterns woven with island heritage and North Atlantic cultural tradition",
    "Shetland wool border in sheep white thread, textile patterns hand-woven with island heritage and Scottish cultural tradition",
    "Orkney stone border in ancient gray yarn, megalithic-inspired patterns woven with prehistoric heritage and Scottish island culture",
    "Manx triskelion border in Celtic silver thread, three-leg patterns hand-woven with Isle of Man heritage and Celtic-Norse cultural fusion",
    "Cornish tin border in metallic gray yarn, mining-inspired patterns woven with Celtic heritage and southwestern English cultural tradition",
    "Breton sailor border in maritime blue thread, nautical patterns hand-woven with Celtic heritage and French coastal cultural tradition",
    "Basque geometric border in cultural red yarn, mathematical patterns woven with Euskera heritage and Pyrenean cultural independence",
    "Catalan modernist border in artistic gold thread, Art Nouveau patterns hand-woven with regional heritage and Mediterranean cultural renaissance"
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
    "Butterfly wing symmetry textile in iridescent blue threads, perfect bilateral reflection pattern hand-woven with lepidoptera wing-inspired design",
    "Rorschach inkblot reflection carpet in deep black wool, psychological symmetry pattern hand-knotted with mirror inkblot-inspired design",
    "Kaleidoscope crystal reflection textile in rainbow prism threads, multi-axis symmetry pattern woven with optical instrument-inspired geometry",
    "Snowflake hexagonal symmetry carpet in ice crystal white wool, six-fold reflection pattern hand-knotted with winter precipitation-inspired design",
    "Peacock feather eye reflection textile in emerald green threads, radial symmetry pattern woven with bird plumage-inspired design",
    "Mandala mirror meditation carpet in spiritual gold wool, contemplative symmetry pattern hand-knotted with Buddhist reflection-inspired geometry",
    "Art Deco fan reflection textile in metallic silver threads, angular symmetry pattern woven with 1920s decorative-inspired geometry",
    "Gothic rose window reflection carpet in cathedral-colored wool, sacred symmetry pattern hand-knotted with stained glass-inspired design",
    "Islamic tile reflection textile in mosque blue threads, mathematical symmetry pattern woven with geometric architecture-inspired design",
    "Chinese paper cut reflection carpet in lucky red wool, bilateral symmetry pattern hand-knotted with folk art-inspired design",
    "Japanese origami crane reflection textile in pure white threads, fold symmetry pattern woven with paper art-inspired geometry",
    "Native American thunderbird reflection carpet in eagle-colored wool, spiritual symmetry pattern hand-knotted with totemic-inspired design",
    "Egyptian scarab beetle reflection textile in pharaonic gold threads, sacred symmetry pattern woven with ancient Egyptian-inspired symbolism",
    "Celtic triple spiral reflection carpet in ancient green wool, rotational symmetry pattern hand-knotted with druidic-inspired geometry",
    "Hindu yantra reflection textile in sacred saffron threads, divine symmetry pattern woven with tantric-inspired sacred geometry",
    "Aztec feathered serpent reflection carpet in obsidian black wool, mythological symmetry pattern hand-knotted with Quetzalcoatl-inspired design",
    "Persian carpet reflection textile in royal crimson threads, textile symmetry pattern woven with Middle Eastern garden paradise-inspired design",
    "Russian matryoshka reflection carpet in folk-colored wool, nesting symmetry pattern hand-knotted with wooden doll-inspired design",
    "African mask reflection textile in tribal earth-toned threads, ceremonial symmetry pattern woven with indigenous-inspired design",
    "Maori tattoo reflection carpet in traditional black wool, cultural symmetry pattern hand-knotted with tā moko-inspired design",
    "Viking shield reflection textile in warrior-colored threads, heraldic symmetry pattern woven with Norse-inspired protection design",
    "Venetian carnival mask reflection carpet in festival gold wool, theatrical symmetry pattern hand-knotted with masquerade-inspired design",
    "Mexican Talavera reflection textile in ceramic blue threads, pottery symmetry pattern woven with colonial-inspired artistic design",
    "Portuguese azulejo reflection carpet in tile blue wool, architectural symmetry pattern hand-knotted with ceramic-inspired design",
    "Dutch delft reflection textile in porcelain white threads, painted symmetry pattern woven with ceramic-inspired design",
    "German cuckoo clock reflection carpet in Black Forest green wool, mechanical symmetry pattern hand-knotted with timepiece-inspired design",
    "Swiss snowflake reflection textile in Alpine white threads, crystalline symmetry pattern woven with winter mountain-inspired design",
    "Austrian crystal reflection carpet in transparent clarity wool, faceted symmetry pattern hand-knotted with glass-inspired design",
    "Czech Bohemian glass reflection textile in rainbow spectrum threads, optical symmetry pattern woven with crystal-inspired design",
    "Polish folk paper cut reflection carpet in traditional red wool, peasant symmetry pattern hand-knotted with wycinanki-inspired design",
    "Hungarian embroidery reflection textile in folk-colored threads, textile symmetry pattern woven with traditional Magyar-inspired design",
    "Romanian wool reflection carpet in Carpathian-colored wool, weaving symmetry pattern hand-knotted with traditional mountain-inspired design",
    "Bulgarian rose reflection textile in Valley of Roses pink threads, floral symmetry pattern woven with cultivation-inspired design",
    "Serbian Orthodox cross reflection carpet in Byzantine gold wool, religious symmetry pattern hand-knotted with sacred-inspired design",
    "Croatian checkerboard reflection textile in national-colored threads, heraldic symmetry pattern woven with shield-inspired design",
    "Slovenian beehive reflection carpet in honey gold wool, agricultural symmetry pattern hand-knotted with apiary-inspired design",
    "Slovakian folk reflection textile in mountain-colored threads, traditional symmetry pattern woven with peasant-inspired design",
    "Lithuanian amber reflection carpet in Baltic gold wool, geological symmetry pattern hand-knotted with fossil resin-inspired design",
    "Latvian folk reflection textile in forest-colored threads, traditional symmetry pattern woven with rural Baltic-inspired design",
    "Estonian folk reflection carpet in coastal-colored wool, maritime symmetry pattern hand-knotted with island-inspired design",
    "Finnish Karelian reflection textile in subarctic-colored threads, regional symmetry pattern woven with forest-inspired design",
    "Norwegian stave church reflection carpet in timber brown wool, architectural symmetry pattern hand-knotted with wooden-inspired design",
    "Swedish Dalarna horse reflection textile in Falu red threads, folk symmetry pattern woven with carved-inspired design",
    "Danish hygge reflection carpet in cozy-colored wool, lifestyle symmetry pattern hand-knotted with comfort-inspired design",
    "Icelandic saga reflection textile in volcanic-colored threads, literary symmetry pattern woven with epic-inspired design",
    "Faroese knitting reflection carpet in wool-colored wool, textile symmetry pattern hand-knotted with fisherman-inspired design",
    "Greenlandic inuksuk reflection textile in arctic stone threads, navigational symmetry pattern woven with cairn-inspired design",
    "Sami reindeer reflection carpet in aurora-colored wool, nomadic symmetry pattern hand-knotted with herding-inspired design",
    "Siberian shaman reflection textile in tundra-colored threads, spiritual symmetry pattern woven with vision-inspired design",
    "Mongolian yurt reflection carpet in steppe-colored wool, dwelling symmetry pattern hand-knotted with nomadic architecture-inspired design"
  ],
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
        
    
    # Single resolution for this script
    TARGET_RESOLUTION = (TARGET_WIDTH, TARGET_HEIGHT)
    GEN_RESOLUTION = (GEN_WIDTH, GEN_HEIGHT)
    
    # Configuration for 20k images (10 classes × 50 prompts × 10 styles × 4 seeds = 20,000)
    config = {
        "output_folder": "/data/generated_carpets_60x80",
        "seeds_per_combination": 4,  # 4 seeds per prompt+style combination
        "target_resolution": TARGET_RESOLUTION,
        "generation_resolution": GEN_RESOLUTION,
        "guidance_scale": 2.5,    # Lower guidance for tiny resolutions
        "num_inference_steps": 20,  # Fewer steps for tiny resolutions
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
    
    # Calculate total target images: 10 classes × 50 prompts × 10 styles × 4 seeds = 20,000
    total_classes = len(prompts_data)
    total_prompts_per_class = len(list(prompts_data.values())[0])  # Should be 50
    total_styles = len(artistic_styles)  # Should be 10
    seeds_per_combo = config["seeds_per_combination"]
    total_target_images = total_classes * total_prompts_per_class * total_styles * seeds_per_combo
    
    print(f"Starting 60x80 carpet generation for {total_classes} classes")
    print(f"Target resolution: {config['target_resolution']}")
    print(f"Generation resolution: {config['generation_resolution']}")
    print(f"Prompts per class: {total_prompts_per_class}")
    print(f"Artistic styles: {total_styles}")
    print(f"Seeds per combination: {seeds_per_combo}")
    print(f"Total target images: {total_target_images}")
    print(f"Guidance scale: {config['guidance_scale']}")
    print(f"Inference steps: {config['num_inference_steps']}")
    
    # Process single resolution (60x80)
    gen_width, gen_height = config["generation_resolution"]
    target_width, target_height = config["target_resolution"]
    resolution_name = f"{target_width}x{target_height}"
    
    print(f"\n{'='*80}")
    print(f"Processing resolution: {resolution_name}")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = Path(config["output_folder"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each class
    for class_name, class_prompts in prompts_data.items():
        print(f"\n{'-'*60}")
        print(f"Processing class: {class_name.upper()}")
        print(f"Prompts in class: {len(class_prompts)}")
        expected_images_per_class = len(class_prompts) * len(artistic_styles) * seeds_per_combo
        print(f"Expected images per class: {expected_images_per_class}")
        print(f"{'-'*60}")
        
        # Create class directory
        class_dir = output_dir / class_name
        os.makedirs(class_dir, exist_ok=True)
    
        class_successful = 0
        class_failed = 0
        image_counter = 0
        
        # Generate images: 50 prompts × 10 styles × 4 seeds = 2000 per class
        for prompt_idx, prompt in enumerate(class_prompts):
            for style_idx, artistic_style in enumerate(artistic_styles):
                for seed_idx in range(seeds_per_combo):
                    image_counter += 1
                    
                    print(f"\nGenerating image {image_counter}/{expected_images_per_class} for {class_name}")
                    print(f"Prompt {prompt_idx + 1}/{len(class_prompts)}: {prompt[:60]}...")
                    print(f"Style {style_idx + 1}/{len(artistic_styles)}: {artistic_style[:40]}...")
                    print(f"Seed {seed_idx + 1}/{seeds_per_combo}")
                    
                    try:
                        # Generate deterministic seed based on indices
                        base_seed = hash(f"{class_name}_{prompt_idx}_{style_idx}_{seed_idx}") % (2**32)
                        seed = abs(base_seed)
                        
                        # Simplified prompt for tiny resolutions - remove seamless techniques
                        enhanced_prompt = (
                            f"{prompt}, {artistic_style}, pattern design, "
                            "flat design, vector art style, graphic design, clean lines, "
                            "solid colors, high contrast, decorative motif, "
                            "no shadows, no 3D effects, flat illustration"
                        )
                        
                        print(f"  Generating with seed: {seed}, guidance: {config['guidance_scale']:.2f}")
                        
                        # Generate image at 16-divisible resolution
                        image, actual_seed = generator.generate_carpet_image(
                            prompt=enhanced_prompt,
                            negative_prompt=config["base_negative_prompt"],
                            width=gen_width,
                            height=gen_height,
                            guidance_scale=config["guidance_scale"],
                            num_inference_steps=config["num_inference_steps"],
                            seed=seed,
                            enable_seamless=False  # Disabled for small resolutions
                        )
                        
                        # Resize to target resolution
                        if (gen_width, gen_height) != (target_width, target_height):
                            image = image.resize((target_width, target_height), Image.LANCZOS)
                    
                        # Save image with detailed naming
                        filename = f"{resolution_name}_{class_name}_{image_counter:05d}_p{prompt_idx + 1:02d}_s{style_idx + 1:02d}_seed{seed_idx + 1}_{actual_seed}.png"
                        image_path = class_dir / filename
                        image.save(image_path, "PNG", quality=95)
                        
                        print(f"  ✓ Saved: {filename}")
                        
                        class_successful += 1
                        successful_generations += 1
                        total_images += 1
                        
                        # Clear GPU memory
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"  ✗ Failed to generate image {image_counter} for {class_name}: {str(e)}")
                        class_failed += 1
                        failed_generations += 1
                        total_images += 1
                        continue
        
        print(f"\nClass {class_name} complete:")
        print(f"  Successful: {class_successful}/{expected_images_per_class}")
        print(f"  Failed: {class_failed}")
        if (class_successful + class_failed) > 0:
            print(f"  Success rate: {(class_successful/(class_successful + class_failed)*100):.1f}%")
    
    print(f"\nAll classes complete for {resolution_name}.")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Overall success rate: {(successful_generations/total_images*100):.1f}%")
    print(f"Generation mode: {seeds_per_combo} seeds per prompt+style combination")
    print(f"Resolution processed: {resolution_name}")
    print(f"Total classes: {len(prompts_data)}")
    print(f"Total combinations: {total_prompts_per_class * total_styles * total_classes}")
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
            "total_prompts_per_class": total_prompts_per_class,
            "total_styles": total_styles,
            "seeds_per_combination": seeds_per_combo,
            "generation_mode": f"{seeds_per_combo}_seeds_per_combination"
        },
        "model_info": {
            "model_id": MODEL_ID,
            "guidance_scale": config["guidance_scale"],
            "num_inference_steps": config["num_inference_steps"],
            "target_resolution": config["target_resolution"],
            "generation_resolution": config["generation_resolution"]
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
        generate_carpet_dataset_60x80.remote()