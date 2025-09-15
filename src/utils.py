import os
from PIL import Image
from datetime import datetime, timezone

def get_image_metadata(file_path):
    stat_info = os.stat(file_path)
    size = stat_info.st_size   # (bytes)
    uploaded_at = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
    
    with Image.open(file_path) as img:
        width, height = img.size
        format = img.format.lower()  # jpg, png, webp,...

    return {
        "file_name": os.path.basename(file_path),
        "url": os.path.abspath(file_path),
        "uploaded_at": uploaded_at.isoformat(),
        "size": size,
        "format": format,
        "width": width,
        "height": height
    }
    