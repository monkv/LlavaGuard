"""Shared label parsing, image encoding, and resumable dataset file handling."""

import ast
import base64
import io
import json
import math
import os
from pathlib import Path
import tempfile

from llavaguard.taxonomy.PEGI.PEGI_Graph import policy_graph

NA_LABEL = "NA: None applying"


def parse_labels(value):
    """Parse lists, serialized lists, and comma-separated labels; omit NA."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        labels = value
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith('['):
            labels = ast.literal_eval(value)
            if not isinstance(labels, list):
                raise ValueError("Expected a list of label strings")
        else:
            labels = value.split(',')
    else:
        raise ValueError(f"Unsupported labels: {value!r}")
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("Labels must be strings")
    return list(dict.fromkeys(label.strip() for label in labels if label.strip() and label.strip() != NA_LABEL))


def labels_to_string(value):
    """Serialize an empty label set using the canonical NA label."""
    return ', '.join(parse_labels(value)) or NA_LABEL


def sample_labels(sample):
    """Validate subcategories and derive their parent categories."""
    if 'subcategories' not in sample or sample['subcategories'] is None:
        raise ValueError(f"Missing subcategories for {sample.get('id')}")
    labels = parse_labels(sample['subcategories'])
    parents = {sub: (cat, info) for cat, subs in policy_graph.items() for sub, info in subs.items()}
    unknown = set(labels) - set(parents)
    if unknown:
        raise ValueError(f"Unknown subcategories: {sorted(unknown)}")
    if any(parents[label][1]['rating'] == 'Illegal' for label in labels):
        raise ValueError("Illegal-content labels require separate handling from PEGI age ratings")
    categories = list(dict.fromkeys(parents[label][0] for label in labels))
    return labels_to_string(categories), labels


def image_path_of(sample):
    """Read an image path from any supported dataset schema."""
    path = sample.get('image path') or sample.get('image_path') or sample.get('image')
    if not path:
        raise ValueError(f"Missing image path for {sample.get('id')}")
    return path


def pil_to_data_url(image):
    """Encode an image as a JPEG data URL without importing the labeler."""
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def load_records(path, optional=False):
    """Load a record list and reject corrupt data instead of overwriting it."""
    path = Path(path)
    if optional and not path.exists():
        return []
    with path.open(encoding='utf-8') as handle:
        records = json.load(handle)
    if not isinstance(records, list) or any(not isinstance(record, dict) or record.get('id') is None for record in records):
        raise ValueError(f"Expected records with IDs in {path}")
    ids = [str(record['id']) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate record IDs in {path}")
    return records


def write_records(path, records):
    """Save a complete JSON checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(records, handle, ensure_ascii=False, indent=4)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def prepare_build(input_file, output_file, batch_size):
    """Validate paths and resume data before making model requests."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if Path(input_file).resolve() == Path(output_file).resolve():
        raise ValueError('Input and output files must be different')
    return load_records(input_file), load_records(output_file, optional=True)
