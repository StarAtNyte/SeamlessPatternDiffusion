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