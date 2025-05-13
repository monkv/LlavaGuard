import os

local_data_dir = '/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo'


local_image_dirs = {
    'smid':      os.path.join(local_data_dir, 'data', 'images', 'smid'),
    'crawled':   os.path.join(local_data_dir, 'data', 'images', 'crawled_data'),
    'synthetic': os.path.join(local_data_dir, 'data', 'images', 'synthetic_data'),
}

local_annotation_dir = os.path.join(
    local_data_dir,
    'annotations',
    'auto_generated_annotations'
)

