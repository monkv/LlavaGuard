"""Split by image before balancing training data to prevent image leakage."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


from llavaguard.taxonomy.PEGI.PEGI_Graph import get_content_categories, get_max_pegi_score
from pegiguard.data.dataset_utils import parse_labels

def parse_subcategories(value):
    """Parse both list and string subcategories without losing existing lists."""
    return parse_labels(value)


def build_image_level_df(df, img_col="image path", subcat_col="subcategories"):
    """Keep all age variants of each image in the same dataset split."""
    if img_col not in df:
        img_col = "image"
    if img_col not in df or df[img_col].isna().any():
        raise ValueError("Every record must have an image path")
    grouped = df.groupby(img_col, sort=True).agg({
        subcat_col: lambda values: sorted({label for value in values for label in parse_subcategories(value)})
    }).reset_index()
    grouped['age_rating'] = grouped[subcat_col].apply(get_max_pegi_score)
    return grouped, img_col, subcat_col


def multilabel_stratified_train_val(
    img_df,
    img_col,
    subcat_col,
    val_size=0.1,
    random_state=42,
):
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    def build_split_labels(row):
        labels = list(row[subcat_col])
        labels.append(f"PEGI_{row['age_rating']}")
        return labels

    img_df = img_df.copy()
    img_df["split_labels"] = img_df.apply(build_split_labels, axis=1)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(img_df["split_labels"])

    X_dummy = np.zeros((len(img_df), 1))

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(msss.split(X_dummy, Y))

    img_ids = img_df[img_col].values

    return img_ids[train_idx], img_ids[val_idx], Y


def rebalance_train_by_rating_and_rare_subcats(
    train_df,
    img_col="image path",
    label_col="age_rating",
    subcat_col="subcategories",
    target_counts=None,
    min_subcat_count=200,
    alpha=5.0,
    random_state=42,
):
    """Sample images by content rating, preserving all rows for each sampled image."""
    if target_counts is None:
        target_counts = {rating: 2500 for rating in (3, 7, 12, 16, 18)}
    if min_subcat_count < 0 or alpha < 0 or any(target < 0 for target in target_counts.values()):
        raise ValueError("Sampling counts and weights must be nonnegative")
    images, _, _ = build_image_level_df(train_df, img_col, subcat_col)
    images = images.rename(columns={'age_rating': label_col}) if label_col != 'age_rating' else images
    counts = {}
    for labels in images[subcat_col]:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    def rarity_weight(labels):
        return 1.0 + sum(alpha * min_subcat_count / counts[label]
                         for label in labels if counts[label] < min_subcat_count)

    rng = np.random.default_rng(random_state)
    sampled_parts = []
    for rating, target in target_counts.items():
        group = images[images[label_col] == int(rating)]
        if group.empty:
            if target:
                raise ValueError(f"No training images for PEGI {rating}")
            continue
        weights = group[subcat_col].apply(rarity_weight).to_numpy(dtype=float)
        sampled = rng.choice(group[img_col].to_numpy(), size=target,
                             replace=len(group) < target, p=weights / weights.sum())
        sampled_parts.extend(train_df[train_df[img_col] == image] for image in sampled)
    if not sampled_parts:
        return train_df.iloc[:0].copy()
    return pd.concat(sampled_parts, ignore_index=True)


def make_balanced_test_split(
    img_df,
    img_col="image path",
    label_col="age_rating",
    subcat_col="subcategories",
    n_per_class=100,
    min_per_subcat=2,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    img_df = img_df.copy()

    # Ensure list subcategories.
    img_df[subcat_col] = img_df[subcat_col].apply(
        lambda x: x if isinstance(x, list) else parse_subcategories(x)
    )

    selected_indices = set()

    # Guarantee at least min_per_subcat images per subcategory.
    all_subcats = get_content_categories()

    for subcat in all_subcats:
        if subcat == "46. Child Endangerment" or subcat == "47. Child Sexual Abuse":
            continue
        # Find all images containing that category.
        candidates = img_df[
            img_df[subcat_col].apply(lambda cats: subcat in cats)
        ]

        if len(candidates) < min_per_subcat:
            raise ValueError(
                f"Subcategory '{subcat}' has only {len(candidates)} images, "
                f"but requested {min_per_subcat}."
            )

        # Which candidates are already in the test set.
        already_selected = candidates[candidates.index.isin(selected_indices)]

        need = min_per_subcat - len(already_selected)

        if need > 0:
            # Remove already-selected candidates.
            remaining = candidates[~candidates.index.isin(selected_indices)]
            # Randomly choose missing images.
            chosen = rng.choice(
                remaining.index.to_numpy(),
                size=need,
                replace=False,
            )

            selected_indices.update(chosen)

    # Fill each PEGI rating up to n_per_class.
    for rating, group in img_df.groupby(label_col):
        already = group[group.index.isin(selected_indices)]
        need = n_per_class - len(already)

        if need < 0:
            raise ValueError(f"Subcategory coverage needs more than {n_per_class} images for PEGI {rating}")
        if need == 0:
            continue

        remaining = group[~group.index.isin(selected_indices)]

        if len(remaining) < need:
            raise ValueError(
                f"PEGI {rating} needs {need} more images, "
                f"but only {len(remaining)} available."
            )

        chosen = rng.choice(
            remaining.index.to_numpy(),
            size=need,
            replace=False,
        )

        selected_indices.update(chosen)

    test_df_img = img_df.loc[sorted(selected_indices)].copy()
    test_imgs = test_df_img[img_col].values

    remaining_img_df = img_df[~img_df[img_col].isin(test_imgs)].copy()

    return remaining_img_df, test_imgs


def spread_train_order(
    train_df,
    label_col="age_rating",
    img_col="image path",
    random_state=42,
    max_recent=50,
):

    if max_recent < 1:
        raise ValueError("max_recent must be positive")
    if train_df.empty:
        return train_df.copy()
    rng = np.random.default_rng(random_state)

    groups = {
        label: group.sample(frac=1, random_state=random_state).to_dict("records") # Take 100% of the rows (frac=1) in random order.
        for label, group in train_df.groupby(label_col)
    } # Creates queues of rows for each label, shuffled in random order.

    ordered = []
    recent_imgs = []

    while any(groups.values()):
        labels = [l for l, rows in groups.items() if rows]
        rng.shuffle(labels)

        picked = False

        for label in labels:
            rows = groups[label]

            # Prefer a sample whose image was not used recently.
            for i, row in enumerate(rows):
                if row[img_col] not in recent_imgs:
                    ordered.append(row)
                    recent_imgs.append(row[img_col])
                    recent_imgs = recent_imgs[-max_recent:]
                    rows.pop(i)
                    picked = True
                    break

            if picked:
                break

        # Fallback if all remaining images are recent duplicates.
        if not picked:
            label = labels[0]
            row = groups[label].pop(0)
            ordered.append(row)
            recent_imgs.append(row[img_col])
            recent_imgs = recent_imgs[-max_recent:]

    return pd.DataFrame(ordered)


if __name__ == "__main__":
    from llavaguard_config import local_data_dir
    TEMPLATE_VERSION = "v2"
    MODE = "age_rating"


    file_path = f"{local_data_dir}/datasets/PEGI-Guard-DS/v4/{MODE}/all_data.json"
    print(f"** Loading data from {file_path} **")

    ds_out = f"{local_data_dir}/datasets/PEGI-Guard-DS/{MODE}/{TEMPLATE_VERSION}"
    print(f"** Output dataset will be saved to {ds_out} **")

    df = pd.read_json(file_path, orient="records")

    # Ensure subcategories is a list (parse if string).
    df['subcategories'] = df['subcategories'].apply(parse_subcategories)


    img_df, img_col, subcat_col = build_image_level_df(df)
    # Use content ratings consistently for both age-rating and safety records.
    df['age_rating'] = df[img_col].map(img_df.set_index(img_col)['age_rating'])

    # Parameters for test split and train rebalancing.
    target_counts={
        3: 2200,
        7: 2200,
        12: 2200,
        16: 2200,
        18: 2200,
    }

    label_col="age_rating"
    min_subcat_count=100
    alpha=5.0
    random_state=42

    # Create balanced test set: n_per_class images per label_col class.
    img_df_remaining, test_imgs = make_balanced_test_split(
        img_df,
        img_col=img_col,
        label_col=label_col,
        subcat_col=subcat_col,
        n_per_class=100,
        min_per_subcat=2,
        random_state=42,
    )

    val_size=0.1
    train_imgs, val_imgs, Y = multilabel_stratified_train_val(
        img_df_remaining,
        img_col=img_col,
        subcat_col=subcat_col,
        val_size=val_size,
        random_state=42,
    )

    train_df = df[df[img_col].isin(train_imgs)].copy()
    val_df   = df[df[img_col].isin(val_imgs)].copy()
    test_df  = df[df[img_col].isin(test_imgs)].copy()

    train_df = rebalance_train_by_rating_and_rare_subcats(
        train_df,
        img_col,
        label_col,
        subcat_col,
        target_counts,
        min_subcat_count,
        alpha,
        random_state,
    )

    train_df = spread_train_order(
        train_df,
        label_col,
        img_col,
        random_state,
        max_recent=50,
    )

    splits = [set(part[img_col]) for part in (train_df, val_df, test_df)]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("An image appears in multiple splits")
    Path(ds_out).mkdir(parents=True, exist_ok=True)
    train_df.to_json(f"{ds_out}/train.json", orient="records", indent=2)
    val_df.to_json(  f"{ds_out}/val.json",   orient="records", indent=2)
    test_df.to_json( f"{ds_out}/test.json",  orient="records", indent=2)

    # ---------------------------------------------------------------------.
    # Save dataset info + config.
    # ---------------------------------------------------------------------.


    min_subcat_count = 100
    alpha = 5.0
    random_state = 42
    n_per_class = 100
    min_per_subcat = 2

    all_subcat_counts = (
        df[subcat_col]
        .explode()
        .value_counts()
    )

    train_subcat_counts = (
        train_df[subcat_col]
        .explode()
        .value_counts()
    )

    val_subcat_counts = (
        val_df[subcat_col]
        .explode()
        .value_counts()
    )

    test_subcat_counts = (
        test_df[subcat_col]
        .explode()
        .value_counts()
    )

    print_txt = f'''
    #################################################################################################
    DS Info:
    Prepared PEGI-Guard dataset for instruction tuning from dir {file_path}
    Dataset output dir: {ds_out}
    Version: {TEMPLATE_VERSION}, Mode: {MODE}
    Split strategy: balanced on age-rating test split + multilabel stratified train/val split + oversampling of rare subcategories in train
    Oversampling: rare-subcategory weighting
    Random state: {random_state}
    #################################################################################################
    All data: {len(df)} unique images: {df[img_col].nunique()}
    Train data: {len(train_df)} unique images: {train_df[img_col].nunique()}
    Eval data: {len(val_df)} unique images: {val_df[img_col].nunique()}
    Test data: {len(test_df)} unique images: {test_df[img_col].nunique()}

    PEGI distribution all:
    {df[label_col].value_counts().sort_index()}

    PEGI distribution train:
    {train_df[label_col].value_counts().sort_index()}

    PEGI distribution eval:
    {val_df[label_col].value_counts().sort_index()}

    PEGI distribution test:
    {test_df[label_col].value_counts().sort_index()}

    Subcategory distribution all:
    {all_subcat_counts}

    Subcategory distribution train:
    {train_subcat_counts}

    Subcategory distribution eval:
    {val_subcat_counts}

    Subcategory distribution test:
    {test_subcat_counts}
    #################################################################################################
    Test split config:
    n_per_class: {n_per_class}
    min_per_subcat: {min_per_subcat}

    Train rebalance config:
    target_counts: {target_counts}
    min_subcat_count: {min_subcat_count}
    alpha: {alpha}
    #################################################################################################
    Train Data saved at: {ds_out}/train.json
    Eval Data saved at: {ds_out}/val.json
    Test Data saved at: {ds_out}/test.json
    #################################################################################################
    '''

    print(print_txt)

    with open(f"{ds_out}/ds_info.txt", "w") as f:
        f.write(print_txt)

    ds_config = {
        "version": TEMPLATE_VERSION,
        "mode": MODE,
        "input_file": file_path,
        "output_dir": ds_out,
        "split_strategy": {
            "test": {
                "type": "balanced_pegi_with_min_subcategory_coverage",
                "n_per_class": n_per_class,
                "min_per_subcat": min_per_subcat,
            },
            "train_val": {
                "type": "multilabel_stratified_shuffle_split",
                "val_size": val_size,
            },
        },
        "rebalance_strategy": {
            "type": "PEGI_age_rating_target_counts_with_rare_subcategory_weighting",
            "target_counts": target_counts,
            "min_subcat_count": min_subcat_count,
            "alpha": alpha,
        },
        "ordering_strategy": {
            "type": "spread_train_order",
            "disable_shuffling_required": True,
            "llama_factory_argument": "disable_shuffling: true",
        },
        "random_state": random_state,
        "columns": {
            "image": img_col,
            "label": label_col,
            "subcategories": subcat_col,
        },
        "breakpoints": [1000, 5000, 7500, 10000, 11000],
    }

    with open(f"{ds_out}/ds_config.txt", "w") as f:
        json.dump(ds_config, f, indent=4)

    # Save splits.
    print(f"\nFinal sizes : train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"\nDatasets written to:\n{ds_out}/train.json \n{ds_out}/val.json \n{ds_out}/test.json")
