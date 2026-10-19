"""Generate a large, realistic-looking product catalog into products.json.

The 20 hand-written products (P001-P020) are kept as-is; the rest are generated
deterministically (fixed random seed), so re-running gives the same catalog.

    python generate_products.py          # 1000 products
    python generate_products.py 5000     # any size
"""
import json
import random
import sys
from pathlib import Path

PRODUCTS_FILE = Path(__file__).parent / "products.json"
HANDCRAFTED_COUNT = 20
SEED = 42

# Each category: brands, product types (name, tags), model lines, price range,
# selling points used in descriptions, and a spec generator.
CATALOG = {
    "headphones": {
        "brands": ["Sony", "Bose", "Sennheiser", "JBL", "Audio-Technica", "Beats", "Anker", "Jabra", "Skullcandy", "Shure"],
        "types": [
            ("Wireless Over-Ear Headphones", ["wireless", "bluetooth", "over-ear"]),
            ("Noise-Cancelling Headphones", ["wireless", "noise-cancelling", "over-ear", "travel"]),
            ("True Wireless Earbuds", ["wireless", "earbuds", "bluetooth"]),
            ("Sport Earbuds", ["wireless", "earbuds", "sport", "water-resistant"]),
            ("Studio Monitor Headphones", ["wired", "studio", "over-ear"]),
            ("Gaming Headset", ["gaming", "microphone", "over-ear"]),
        ],
        "lines": ["Pro", "Max", "Lite", "Air", "Elite", "Studio", "Go", "Plus", "Neo", "Pulse"],
        "price": (19, 499),
        "features": ["deep, balanced bass", "all-day comfort", "crystal-clear calls", "a long-lasting battery", "fast USB-C charging"],
        "specs": lambda r: {
            "battery_hours": r.choice([6, 8, 20, 24, 30, 40, 60]),
            "weight_g": r.randint(5, 350),
            "connectivity": r.choice(["Bluetooth 5.0", "Bluetooth 5.2", "Bluetooth 5.3", "Wired 3.5mm", "USB-C"]),
        },
    },
    "laptops": {
        "brands": ["Apple", "Dell", "HP", "Lenovo", "ASUS", "Acer", "Microsoft", "MSI", "Razer", "Samsung"],
        "types": [
            ("Ultrabook", ["lightweight", "long-battery", "student"]),
            ("Gaming Laptop", ["gaming", "performance", "rgb"]),
            ("2-in-1 Laptop", ["touchscreen", "convertible", "lightweight"]),
            ("Business Laptop", ["business", "secure", "long-battery"]),
            ("Creator Laptop", ["creator", "performance", "oled"]),
            ("Chromebook", ["chromeos", "budget", "student"]),
        ],
        "lines": ["Nova", "Edge", "Flex", "Vision", "Spark", "Prime", "Orbit", "Volt", "Core", "Stride"],
        "price": (249, 3499),
        "features": ["a vivid high-resolution display", "a backlit keyboard", "Wi-Fi 6E", "a fast SSD", "a premium aluminum build"],
        "specs": lambda r: {
            "ram_gb": r.choice([8, 16, 32, 64]),
            "storage_gb": r.choice([128, 256, 512, 1024, 2048]),
            "screen_in": r.choice([13.3, 14.0, 15.6, 16.0, 17.3]),
            "weight_kg": round(r.uniform(1.0, 3.0), 2),
        },
    },
    "phones": {
        "brands": ["Apple", "Samsung", "Google", "OnePlus", "Motorola", "Xiaomi", "Nothing", "Sony", "Nokia", "Oppo"],
        "types": [
            ("Smartphone", ["5g", "camera"]),
            ("Camera Phone", ["camera", "5g", "photography"]),
            ("Budget Phone", ["budget", "long-battery"]),
            ("Foldable Phone", ["foldable", "5g", "premium"]),
            ("Rugged Phone", ["rugged", "waterproof", "long-battery"]),
        ],
        "lines": ["Nova", "Edge", "Vista", "Ultra", "Mini", "Pro", "Lite", "Flip", "Prime", "Aura"],
        "price": (99, 1899),
        "features": ["an OLED display", "a multi-lens camera system", "all-day battery", "fast wireless charging", "IP68 water resistance"],
        "specs": lambda r: {
            "storage_gb": r.choice([64, 128, 256, 512, 1024]),
            "screen_in": round(r.uniform(5.8, 7.6), 1),
            "battery_mah": r.choice([3500, 4000, 4500, 5000, 6000]),
        },
    },
    "tablets": {
        "brands": ["Apple", "Samsung", "Amazon", "Lenovo", "Microsoft", "Xiaomi", "Huawei"],
        "types": [
            ("Tablet", ["portable", "streaming"]),
            ("Pro Tablet", ["premium", "creator", "stylus"]),
            ("Kids Tablet", ["kids", "durable", "budget"]),
            ("Drawing Tablet", ["stylus", "creator", "art"]),
        ],
        "lines": ["Tab", "Slate", "Canvas", "Air", "Pro", "Mini", "Plus", "Max"],
        "price": (59, 1499),
        "features": ["a laminated display", "quad speakers", "stylus support", "a slim design", "long battery life"],
        "specs": lambda r: {
            "storage_gb": r.choice([32, 64, 128, 256, 512]),
            "screen_in": round(r.uniform(7.0, 13.0), 1),
            "battery_hours": r.choice([8, 10, 12, 15]),
        },
    },
    "cameras": {
        "brands": ["Canon", "Nikon", "Sony", "Fujifilm", "Panasonic", "GoPro", "DJI", "Olympus", "Leica"],
        "types": [
            ("Mirrorless Camera", ["mirrorless", "photography", "4k"]),
            ("DSLR Camera", ["dslr", "photography"]),
            ("Action Camera", ["action", "waterproof", "4k", "travel"]),
            ("Instant Camera", ["instant", "fun", "budget"]),
            ("Vlogging Camera", ["vlogging", "4k", "compact"]),
        ],
        "lines": ["Vision", "Focus", "Aperture", "Pro", "Lens", "Snap", "Frame", "Studio", "Trek"],
        "price": (69, 3999),
        "features": ["fast autofocus", "in-body stabilization", "4K video recording", "weather sealing", "excellent low-light performance"],
        "specs": lambda r: {
            "megapixels": r.choice([12, 20, 24, 26, 33, 45, 61]),
            "video": r.choice(["1080p", "4K30", "4K60", "5.3K60", "8K"]),
            "weight_g": r.randint(120, 950),
        },
    },
    "watches": {
        "brands": ["Apple", "Garmin", "Samsung", "Fitbit", "Amazfit", "Casio", "Fossil", "Withings", "Polar"],
        "types": [
            ("Smartwatch", ["smartwatch", "fitness", "notifications"]),
            ("GPS Running Watch", ["gps", "running", "fitness"]),
            ("Fitness Tracker", ["fitness", "budget", "sleep-tracking"]),
            ("Hybrid Watch", ["hybrid", "classic", "long-battery"]),
            ("Dive Watch", ["waterproof", "outdoor", "durable"]),
        ],
        "lines": ["Pulse", "Active", "Pace", "Trail", "Sport", "Classic", "Ultra", "Fit", "Explorer"],
        "price": (29, 1099),
        "features": ["heart-rate tracking", "built-in GPS", "an always-on display", "multi-day battery", "sleep tracking"],
        "specs": lambda r: {
            "battery_days": r.choice([1, 2, 5, 7, 14, 21]),
            "water_resistance_m": r.choice([30, 50, 100, 200]),
            "case_mm": r.choice([38, 40, 42, 44, 46, 47]),
        },
    },
    "gaming": {
        "brands": ["Sony", "Microsoft", "Nintendo", "Logitech", "Razer", "SteelSeries", "Corsair", "HyperX", "8BitDo"],
        "types": [
            ("Wireless Controller", ["controller", "wireless", "gaming"]),
            ("Mechanical Gaming Keyboard", ["keyboard", "mechanical", "rgb", "gaming"]),
            ("Gaming Mouse", ["mouse", "rgb", "gaming"]),
            ("Gaming Chair", ["chair", "ergonomic", "gaming"]),
            ("Racing Wheel", ["racing", "simulation", "gaming"]),
            ("Handheld Console", ["console", "portable", "gaming"]),
        ],
        "lines": ["Pro", "Elite", "Striker", "Phantom", "Titan", "Raptor", "Nova", "Vortex", "Ultimate"],
        "price": (19, 699),
        "features": ["low-latency wireless", "customizable RGB lighting", "programmable buttons", "pro-grade sensors", "a durable build"],
        "specs": lambda r: {
            "connectivity": r.choice(["Wired USB", "2.4GHz wireless", "Bluetooth", "Wireless + Bluetooth"]),
            "weight_g": r.randint(60, 2500),
            "warranty_years": r.choice([1, 2, 3]),
        },
    },
    "electronics": {
        "brands": ["Samsung", "LG", "Logitech", "Amazon", "Anker", "TP-Link", "Belkin", "Dell", "Sonos", "Roku"],
        "types": [
            ("4K Monitor", ["monitor", "4k", "productivity"]),
            ("Smart Speaker", ["smart-home", "speaker", "voice-assistant"]),
            ("Portable Charger", ["charger", "portable", "travel"]),
            ("Wi-Fi Router", ["router", "wifi", "networking"]),
            ("Wireless Mouse", ["mouse", "wireless", "productivity"]),
            ("Streaming Stick", ["streaming", "4k", "tv"]),
            ("E-Reader", ["e-reader", "travel", "long-battery"]),
        ],
        "lines": ["Smart", "Ultra", "Plus", "Max", "Mini", "Pro", "Connect", "Wave", "Link"],
        "price": (15, 1299),
        "features": ["easy setup", "energy-efficient design", "voice control support", "a compact footprint", "a 2-year warranty"],
        "specs": lambda r: {
            "connectivity": r.choice(["Wi-Fi", "Bluetooth", "USB-C", "Wi-Fi + Bluetooth", "HDMI"]),
            "power_w": r.choice([5, 10, 20, 45, 65, 100]),
        },
    },
    "home": {
        "brands": ["Dyson", "iRobot", "Philips", "Shark", "Levoit", "Ring", "Nest", "Eufy", "Honeywell", "IKEA"],
        "types": [
            ("Robot Vacuum", ["robot", "cleaning", "smart-home"]),
            ("Cordless Vacuum", ["cordless", "cleaning", "lightweight"]),
            ("Air Purifier", ["air-purifier", "allergy", "quiet"]),
            ("Smart Thermostat", ["smart-home", "energy-saving"]),
            ("Video Doorbell", ["security", "smart-home", "camera"]),
            ("LED Desk Lamp", ["lighting", "dimmable", "office"]),
            ("Humidifier", ["humidifier", "quiet", "bedroom"]),
        ],
        "lines": ["Core", "Clean", "Pure", "Smart", "Breeze", "Halo", "Home", "Fresh", "Glow"],
        "price": (19, 899),
        "features": ["app control", "whisper-quiet operation", "a HEPA filter", "smart scheduling", "a sleek modern design"],
        "specs": lambda r: {
            "power_w": r.choice([5, 10, 25, 40, 100, 350, 600]),
            "noise_db": r.randint(20, 75),
            "smart_home": r.choice(["Alexa", "Google Home", "Apple HomeKit", "Alexa + Google", "None"]),
        },
    },
    "kitchen": {
        "brands": ["Ninja", "Breville", "KitchenAid", "Instant Pot", "Cuisinart", "Vitamix", "Keurig", "De'Longhi", "Lodge", "OXO"],
        "types": [
            ("Air Fryer", ["air-fryer", "healthy"]),
            ("Espresso Machine", ["coffee", "espresso"]),
            ("Stand Mixer", ["baking", "mixer"]),
            ("High-Speed Blender", ["blender", "smoothies"]),
            ("Multi-Cooker", ["pressure-cooker", "multi-cooker", "family"]),
            ("Cast Iron Skillet", ["cookware", "cast-iron", "durable"]),
            ("Electric Kettle", ["kettle", "tea", "fast"]),
        ],
        "lines": ["Pro", "Max", "Classic", "Duo", "Chef", "Barista", "Artisan", "Express", "Smart"],
        "price": (19, 999),
        "features": ["easy cleanup", "multiple preset programs", "a dishwasher-safe design", "a stainless steel finish", "fast heat-up"],
        "specs": lambda r: {
            "capacity_l": round(r.uniform(0.5, 8.0), 1),
            "watts": r.choice([0, 800, 1000, 1200, 1500, 1800]),
        },
    },
    "fitness": {
        "brands": ["Bowflex", "Peloton", "NordicTrack", "Theragun", "Manduka", "Rogue", "TRX", "Hydro Flask", "Garmin"],
        "types": [
            ("Adjustable Dumbbells", ["strength", "home-gym", "weights"]),
            ("Yoga Mat", ["yoga", "non-slip", "eco-friendly"]),
            ("Massage Gun", ["recovery", "massage", "portable"]),
            ("Exercise Bike", ["cardio", "home-gym", "cycling"]),
            ("Resistance Bands Set", ["strength", "portable", "budget"]),
            ("Insulated Water Bottle", ["hydration", "travel", "outdoor"]),
        ],
        "lines": ["Pro", "Elite", "Mini", "Core", "Flex", "Prime", "Studio", "Go"],
        "price": (12, 2499),
        "features": ["commercial-grade materials", "a space-saving design", "quick adjustments", "a lifetime warranty", "easy storage"],
        "specs": lambda r: {
            "weight_kg": round(r.uniform(0.3, 60.0), 1),
            "material": r.choice(["steel", "natural rubber", "TPE", "aluminum", "stainless steel"]),
        },
    },
    "shoes": {
        "brands": ["Nike", "Adidas", "New Balance", "ASICS", "Brooks", "Hoka", "Merrell", "Salomon", "Puma", "Converse"],
        "types": [
            ("Running Shoes", ["running", "cushioned", "daily-trainer"]),
            ("Trail Running Shoes", ["trail", "running", "outdoor"]),
            ("Hiking Boots", ["hiking", "outdoor", "waterproof"]),
            ("Lifestyle Sneakers", ["casual", "everyday", "streetwear"]),
            ("Basketball Shoes", ["basketball", "court", "support"]),
            ("Walking Shoes", ["walking", "comfort", "everyday"]),
        ],
        "lines": ["Cloud", "Swift", "Glide", "Stride", "Trail", "Classic", "Velocity", "Summit", "Flow", "Rush"],
        "price": (39, 279),
        "features": ["responsive cushioning", "a breathable mesh upper", "a grippy outsole", "a secure fit", "lightweight comfort"],
        "specs": lambda r: {
            "drop_mm": r.choice([0, 4, 6, 8, 10, 12]),
            "weight_g": r.randint(200, 650),
        },
    },
    "clothing": {
        "brands": ["Patagonia", "The North Face", "Levi's", "Uniqlo", "Columbia", "Under Armour", "Lululemon", "Carhartt", "Champion"],
        "types": [
            ("Rain Jacket", ["jacket", "waterproof", "outdoor"]),
            ("Fleece Hoodie", ["hoodie", "warm", "casual"]),
            ("Slim Fit Jeans", ["jeans", "denim", "casual"]),
            ("Performance T-Shirt", ["t-shirt", "athletic", "moisture-wicking"]),
            ("Puffer Jacket", ["jacket", "warm", "winter"]),
            ("Yoga Leggings", ["leggings", "athletic", "stretch"]),
        ],
        "lines": ["Classic", "Essential", "Tech", "Heritage", "Trail", "Motion", "Everyday", "Alpine"],
        "price": (14, 399),
        "features": ["a relaxed fit", "recycled materials", "four-way stretch", "a water-repellent finish", "reinforced seams"],
        "specs": lambda r: {
            "sizes": r.choice(["XS-XL", "S-XXL", "S-XL", "28-38 waist"]),
            "material": r.choice(["cotton", "polyester", "nylon", "merino wool", "recycled polyester", "denim"]),
        },
    },
    "beauty": {
        "brands": ["Dyson", "Philips", "Braun", "Olaplex", "CeraVe", "La Roche-Posay", "Foreo", "Remington", "Panasonic"],
        "types": [
            ("Hair Dryer", ["hair", "styling", "fast-drying"]),
            ("Electric Shaver", ["shaving", "grooming", "rechargeable"]),
            ("Electric Toothbrush", ["oral-care", "rechargeable"]),
            ("Facial Cleansing Brush", ["skincare", "cleansing"]),
            ("Moisturizing Cream", ["skincare", "hydrating", "sensitive-skin"]),
            ("Hair Straightener", ["hair", "styling", "ceramic"]),
        ],
        "lines": ["Pro", "Daily", "Silk", "Glow", "Pure", "Radiance", "Gentle", "Ionic"],
        "price": (9, 599),
        "features": ["dermatologist-tested formulas", "multiple heat settings", "a travel-friendly size", "fast charging", "gentle results"],
        "specs": lambda r: {
            "battery_minutes": r.choice([0, 30, 45, 60, 90, 120]),
            "suitable_for": r.choice(["all hair types", "sensitive skin", "all skin types", "men", "women"]),
        },
    },
    "books": {
        "brands": ["Penguin", "HarperCollins", "O'Reilly", "Random House", "Scholastic", "Simon & Schuster", "Hachette", "Wiley"],
        "types": [
            ("Mystery Novel", ["fiction", "mystery", "bestseller"]),
            ("Science Fiction Novel", ["fiction", "sci-fi"]),
            ("Programming Guide", ["non-fiction", "programming", "tech"]),
            ("Cookbook", ["non-fiction", "cooking", "recipes"]),
            ("Self-Help Book", ["non-fiction", "self-help"]),
            ("Children's Picture Book", ["kids", "picture-book"]),
        ],
        "lines": [],
        # Books are named by title (prefix + noun) rather than brand + model line.
        "titles": {
            "Mystery Novel": (["The Silent", "Midnight", "Hidden", "The Lost", "Whispers of the", "Death at the"],
                              ["Harbor", "Witness", "Key", "Lighthouse", "Manor", "Letter", "Orchard"]),
            "Science Fiction Novel": (["Beyond the", "Echoes of the", "The Last", "Children of the", "Dawn of the"],
                                      ["Stars", "Void", "Nebula", "Machine", "Horizon", "Red Planet", "Signal"]),
            "Programming Guide": (["Learning", "Mastering", "Practical", "Effective", "Hands-On", "Fluent"],
                                  ["Python", "JavaScript", "Rust", "Go", "SQL", "Kubernetes", "Machine Learning", "TypeScript"]),
            "Cookbook": (["Simple", "Everyday", "The Joy of", "Weeknight", "Plant-Based", "Salt & Fire:"],
                         ["Baking", "Pasta", "Italian Cooking", "Vegetarian Meals", "Grilling", "Soups", "Desserts"]),
            "Self-Help Book": (["The Power of", "The Art of", "Small Steps to", "Deep", "The Little Book of"],
                               ["Habits", "Focus", "Calm", "Courage", "Discipline", "Kindness", "Work"]),
            "Children's Picture Book": (["The Very Sleepy", "Where's the", "Goodnight,", "The Brave Little", "Ten Tiny"],
                                        ["Bear", "Dragon", "Moon", "Puppy", "Octopus", "Owl", "Penguin"]),
        },
        "price": (7, 79),
        "features": ["gripping storytelling", "practical step-by-step examples", "beautiful illustrations", "expert insights", "an award-winning author"],
        "specs": lambda r: {
            "pages": r.randint(32, 900),
            "format": r.choice(["Hardcover", "Paperback", "Kindle", "Audiobook"]),
        },
    },
    "toys": {
        "brands": ["LEGO", "Hasbro", "Mattel", "Melissa & Doug", "Fisher-Price", "Ravensburger", "Hot Wheels", "Nerf", "Playmobil"],
        "types": [
            ("Building Set", ["building", "creative", "kids"]),
            ("Board Game", ["board-game", "family", "party"]),
            ("Jigsaw Puzzle", ["puzzle", "1000-pieces", "family"]),
            ("Remote Control Car", ["rc", "outdoor", "kids"]),
            ("STEM Robot Kit", ["stem", "educational", "coding"]),
            ("Plush Toy", ["plush", "soft", "toddler"]),
        ],
        "lines": ["City", "Classic", "Junior", "Deluxe", "Adventure", "Space", "Creator", "Mega"],
        "price": (9, 499),
        "features": ["hours of fun", "kid-safe materials", "educational play", "easy assembly", "great gift appeal"],
        "specs": lambda r: {
            "age_range": r.choice(["2+", "4+", "6+", "8+", "10+", "14+", "18+"]),
            "pieces": r.choice([1, 50, 250, 500, 1000, 1500, 3000]),
        },
    },
}

# Per-type price ranges (USD); types not listed fall back to the category range.
TYPE_PRICES = {
    "Wireless Over-Ear Headphones": (49, 399), "Noise-Cancelling Headphones": (79, 499), "True Wireless Earbuds": (19, 299),
    "Sport Earbuds": (19, 199), "Studio Monitor Headphones": (49, 399), "Gaming Headset": (29, 299),
    "Ultrabook": (699, 2199), "Gaming Laptop": (899, 3499), "2-in-1 Laptop": (449, 1799), "Business Laptop": (599, 2299),
    "Creator Laptop": (1199, 3499), "Chromebook": (199, 699),
    "Smartphone": (299, 1199), "Camera Phone": (499, 1499), "Budget Phone": (99, 349), "Foldable Phone": (899, 1899), "Rugged Phone": (199, 699),
    "Tablet": (99, 599), "Pro Tablet": (599, 1499), "Kids Tablet": (59, 199), "Drawing Tablet": (49, 899),
    "Mirrorless Camera": (499, 3999), "DSLR Camera": (399, 2999), "Action Camera": (99, 549), "Instant Camera": (49, 199), "Vlogging Camera": (399, 999),
    "Smartwatch": (99, 899), "GPS Running Watch": (149, 999), "Fitness Tracker": (29, 199), "Hybrid Watch": (99, 399), "Dive Watch": (79, 1099),
    "Wireless Controller": (29, 199), "Mechanical Gaming Keyboard": (49, 249), "Gaming Mouse": (19, 179), "Gaming Chair": (149, 699),
    "Racing Wheel": (99, 599), "Handheld Console": (199, 699),
    "4K Monitor": (249, 1299), "Smart Speaker": (29, 449), "Portable Charger": (15, 129), "Wi-Fi Router": (39, 499),
    "Wireless Mouse": (15, 129), "Streaming Stick": (29, 99), "E-Reader": (99, 399),
    "Robot Vacuum": (149, 899), "Cordless Vacuum": (99, 749), "Air Purifier": (59, 649), "Smart Thermostat": (79, 279),
    "Video Doorbell": (49, 299), "LED Desk Lamp": (19, 149), "Humidifier": (25, 199),
    "Air Fryer": (49, 249), "Espresso Machine": (99, 999), "Stand Mixer": (99, 649), "High-Speed Blender": (49, 599),
    "Multi-Cooker": (59, 249), "Cast Iron Skillet": (19, 89), "Electric Kettle": (19, 149),
    "Adjustable Dumbbells": (99, 549), "Yoga Mat": (19, 129), "Massage Gun": (49, 599), "Exercise Bike": (299, 2499),
    "Resistance Bands Set": (12, 49), "Insulated Water Bottle": (15, 59),
    "Running Shoes": (69, 199), "Trail Running Shoes": (79, 199), "Hiking Boots": (89, 279), "Lifestyle Sneakers": (49, 179),
    "Basketball Shoes": (69, 229), "Walking Shoes": (39, 149),
    "Rain Jacket": (59, 349), "Fleece Hoodie": (29, 149), "Slim Fit Jeans": (29, 129), "Performance T-Shirt": (14, 59),
    "Puffer Jacket": (79, 399), "Yoga Leggings": (25, 129),
    "Hair Dryer": (29, 499), "Electric Shaver": (39, 349), "Electric Toothbrush": (25, 299), "Facial Cleansing Brush": (29, 249),
    "Moisturizing Cream": (9, 69), "Hair Straightener": (25, 299),
    "Mystery Novel": (8, 29), "Science Fiction Novel": (8, 29), "Programming Guide": (29, 79), "Cookbook": (15, 49),
    "Self-Help Book": (10, 32), "Children's Picture Book": (7, 24),
    "Building Set": (19, 499), "Board Game": (15, 79), "Jigsaw Puzzle": (12, 39), "Remote Control Car": (25, 249),
    "STEM Robot Kit": (39, 299), "Plush Toy": (9, 49),
}

ADJECTIVES = ["Premium", "Compact", "Versatile", "Powerful", "Lightweight", "Reliable", "Stylish", "Award-winning", "Top-rated", "Everyday"]


def realistic_price(r: random.Random, low: float, high: float) -> float:
    # Skew toward the cheaper end, like real catalogs.
    value = low + (high - low) * (r.random() ** 1.8)
    ending = r.choice([0.99, 0.99, 0.95, 0.00, 0.49])
    return round(max(int(value) + ending, low), 2)


def make_product(r: random.Random, number: int, category: str, used_names: set) -> dict:
    cfg = CATALOG[category]
    for attempt in range(60):
        brand = r.choice(cfg["brands"])
        type_name, type_tags = r.choice(cfg["types"])
        if "titles" in cfg:
            prefixes, nouns = cfg["titles"][type_name]
            name = f"{r.choice(prefixes)} {r.choice(nouns)} ({type_name})"
        else:
            line = r.choice(cfg["lines"])
            model = r.choice([str(r.randint(2, 9)), str(r.randint(1, 9) * 100), f"{r.randint(1, 9)}{r.choice(['', 'X', 'S', 'i'])}", ""])
            name = " ".join(part for part in [brand, line, model, type_name] if part)
        if attempt == 59:
            name = f"{name} - Edition {number}"
        if name not in used_names:
            used_names.add(name)
            break

    rating = round(min(5.0, max(3.0, r.gauss(4.4, 0.3))), 1)
    stock = 0 if r.random() < 0.12 else r.randint(1, 150)
    extra_tags = [t for t in ["bestseller", "new", "eco-friendly", "gift-idea", "budget", "premium"] if r.random() < 0.08]
    features = r.sample(cfg["features"], 2)
    return {
        "id": f"P{number:03d}",
        "name": name,
        "brand": brand,
        "category": category,
        "price": realistic_price(r, *TYPE_PRICES.get(type_name, cfg["price"])),
        "rating": rating,
        "reviews": int(10 ** r.uniform(1, 5)),
        "stock": stock,
        "tags": list(dict.fromkeys(type_tags + extra_tags)),
        "description": f"{r.choice(ADJECTIVES)} {type_name.lower()} from {brand} with {features[0]} and {features[1]}.",
        "specs": cfg["specs"](r),
    }


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    existing = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
    handcrafted = [p for p in existing if int(p["id"][1:]) <= HANDCRAFTED_COUNT]

    r = random.Random(SEED)
    used_names = {p["name"] for p in handcrafted}
    categories = list(CATALOG)
    products = list(handcrafted)
    for number in range(len(handcrafted) + 1, total + 1):
        products.append(make_product(r, number, categories[number % len(categories)], used_names))

    PRODUCTS_FILE.write_text(json.dumps(products, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for p in products:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    print(f"Wrote {len(products)} products to {PRODUCTS_FILE.name}")
    for name, count in sorted(counts.items()):
        print(f"  {name:<12} {count}")


if __name__ == "__main__":
    main()
