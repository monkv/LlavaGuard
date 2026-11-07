import os

from llavaguard_config import local_data_dir


local_image_dirs = {
    'smid':      os.path.join(local_data_dir, 'images', 'smid'),
    'crawled':   os.path.join(local_data_dir, 'images', 'crawled_data'),
    'synthetic': os.path.join(local_data_dir, 'images', 'synthetic_data'),
}

local_annotation_dir = os.path.join(
    local_data_dir,
    'annotations',
    'auto_generated_annotations'
)

