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
        ]
        }
        
    
    # Configuration
    config = {
        "output_folder": "/data/generated_carpets",
        "images_per_prompt": 10,  # Total images per prompt (2 seeds × 5 styles = 10)
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