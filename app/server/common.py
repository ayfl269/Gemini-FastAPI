import base64
import hashlib
import uuid
from pathlib import Path

from gemini_webapi.constants import Model
from gemini_webapi.types.image import GeneratedImage, Image
from loguru import logger

from app.utils import g_config
from app.utils.helper import detect_image_extension, extract_image_dimensions


def get_model_by_name(name: str) -> Model:
    """Retrieve a Model instance by name."""
    strategy = g_config.gemini.model_strategy
    custom_models = {m.model_name: m for m in g_config.gemini.models if m.model_name}

    if name in custom_models:
        return Model.from_dict(custom_models[name].model_dump())

    if strategy == "overwrite":
        raise ValueError(f"Model '{name}' not found in custom models (strategy='overwrite').")

    return Model.from_name(name)


async def image_to_base64(
    image: Image, temp_dir: Path
) -> tuple[str, int | None, int | None, str, str]:
    """Persist an image provided by gemini_webapi and return base64 plus dimensions, filename, and hash."""
    if isinstance(image, GeneratedImage):
        try:
            saved_path = await image.save(path=str(temp_dir), full_size=True)
        except Exception as e:
            logger.warning(
                f"Failed to download full-size GeneratedImage, retrying with default size: {e}"
            )
            saved_path = await image.save(path=str(temp_dir), full_size=False)
    else:
        saved_path = await image.save(path=str(temp_dir))

    if not saved_path:
        raise ValueError("Failed to save generated image")

    original_path = Path(saved_path)
    data = original_path.read_bytes()
    suffix = original_path.suffix

    if not suffix:
        detected_ext = detect_image_extension(data)
        if detected_ext:
            suffix = detected_ext
        else:
            # Fallback if detection fails
            suffix = ".png" if isinstance(image, GeneratedImage) else ".jpg"

    random_name = f"img_{uuid.uuid4().hex}{suffix}"
    new_path = temp_dir / random_name
    original_path.rename(new_path)

    width, height = extract_image_dimensions(data)
    filename = random_name
    file_hash = hashlib.sha256(data).hexdigest()
    return base64.b64encode(data).decode("ascii"), width, height, filename, file_hash
