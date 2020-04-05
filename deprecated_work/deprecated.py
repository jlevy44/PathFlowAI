def extract_patch_info(
    basename,
    input_dir="./",
    annotations=[],
    threshold=0.5,
    patch_size=224,
    generate_finetune_segmentation=False,
    target_class=0,
    intensity_threshold=100.0,
    target_threshold=0.0,
):
    # from collections import OrderedDict
    # annotations=OrderedDict(annotations)
    # from dask.multiprocessing import get

    # import time
    from dask import dataframe as dd
    import dask.delayed
    import multiprocessing
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon
    from itertools import product
    from os.path import join
    import numpy as np
    from pathflowai.utils import (
        load_dataset,
        npy2da,
        create_purple_mask,
        is_coords_in_box,
        is_valid_patch,
    )
    import pandas as pd

    arr, masks = load_dataset(
        join(input_dir, "{}.zarr".format(basename)),
        join(input_dir, "{}_mask.pkl".format(basename)),
    )
    if "annotations" in masks:
        segmentation = True
        if generate_finetune_segmentation:
            segmentation_mask = npy2da(join(input_dir, "{}_mask.npy".format(basename)))
    else:
        segmentation = False
        # masks=np.load(masks['annotations'])
    # npy_file = join(input_dir,'{}.npy'.format(basename))
    purple_mask = create_purple_mask(arr)
    x_max = float(arr.shape[0])
    y_max = float(arr.shape[1])
    x_steps = int((x_max - patch_size) / patch_size)
    y_steps = int((y_max - patch_size) / patch_size)
    for annotation in annotations:
        try:
            masks[annotation] = (
                [unary_union(masks[annotation])] if masks[annotation] else []
            )
        except:
            masks[annotation] = (
                [MultiPolygon(masks[annotation])] if masks[annotation] else []
            )

    # @pysnooper.snoop("process_line.log")
    def return_line_info(row):
        xs = row["x"]
        ys = row["y"]
        xf = xs + patch_size
        yf = ys + patch_size
        print(basename, xs, ys)
        # if is_valid_patch((purple_mask[xs:xf,ys:yf]>=intensity_threshold).compute(), threshold):#.compute()
        # print(xs,ys, 'valid_patch')
        if segmentation:
            row["annotation"] = "segment"
            # info=[basename,xs,ys,patch_size,'segment']
            seg = segmentation_mask[xs:xf, ys:yf].compute()
            # info=info+
            row.iloc[-target_class:] = [(seg == i).mean() for i in range(target_class)]
            # if generate_finetune_segmentation:
        else:
            row.iloc[-len(annotations) :] = [
                is_coords_in_box(
                    coords=np.array([xs, ys]),
                    patch_size=patch_size,
                    boxes=masks[annotation],
                )
                for annotation in annotations
            ]
            row["annotation"] = annotations[
                row.iloc[-len(annotations) :].argmax()
            ]  # [np.argmax(annotation_areas)]
            # info=[basename,xs,ys,patch_size,main_annotation]+annotation_areas
        # else:
        #     if segmentation:
        #         info = [basename, xs, ys, patch_size, 'NA'] + \
        #             [0. for i in range(target_class)]
        #     else:
        #         info = [basename, xs, ys, patch_size, 'NA'] + \
        #             [0. for i in range(len(annotations))]
        return row  # info

    def seg_line(xs, ys, patch_size, segmentation_mask, target_class):
        xf = xs + patch_size
        yf = ys + patch_size
        seg = segmentation_mask[xs:xf, ys:yf]
        return [(seg == i).mean() for i in range(target_class)]

    def annot_line(xs, ys, patch_size, masks, annotations):
        return [
            is_coords_in_box(
                coords=np.array([xs, ys]),
                patch_size=patch_size,
                boxes=masks[annotation],
            )
            for annotation in annotations
        ]

    patch_info = pd.DataFrame(
        [
            (
                [basename, i * patch_size, j * patch_size, patch_size, "NA"]
                + [0.0] * (target_class if segmentation else len(annotations))
            )
            for i, j in product(range(x_steps + 1), range(y_steps + 1))
        ],
        columns=(
            ["ID", "x", "y", "patch_size", "annotation"]
            + (
                annotations
                if not segmentation
                else list([str(i) for i in range(target_class)])
            )
        ),
    )  # [dask.delayed(return_line_info)(i,j) for (i,j) in product(range(x_steps+1),range(y_steps+1))]
    valid_patches = []
    for xs, ys in patch_info[["x", "y"]].values.tolist():
        valid_patches.append(
            dask.delayed(is_valid_patch)(
                xs, ys, patch_size, purple_mask, intensity_threshold, threshold
            )
        )
    patch_info = patch_info.loc[np.array(dask.compute(valid_patches))]
    area_info = []
    if segmentation:
        patch_info.loc[:, "annotation"] = "segment"
        for xs, ys in patch_info[["x", "y"]].values.tolist():
            area_info.append(
                dask.delayed(seg_line)(
                    xs, ys, patch_size, segmentation_mask, target_class
                )
            )
    else:
        for xs, ys in patch_info[["x", "y"]].values.tolist():
            area_info.append(
                [
                    dask.delayed(is_coords_in_box)(
                        xs, ys, patch_size, masks, annotation
                    )
                    for annotation in annotations
                ]
            )
    patch_info.iloc[:, 6:] = np.array(dask.compute(area_info))
    annot = list(patch_info.iloc[:, 6:])
    patch_info.loc[:, "annotation"] = np.vectorize(
        lambda i: annot[patch_info.iloc[i, 6:].argmax()]
    )(
        np.arange(patch_info.shape[0])
    )  # patch_info[np.arange(target_class).astype(str).tolist()].values.argmax(1).astype(str)
    if 0:
        patch_info = dd.from_pandas(
            patch_info, npartitions=2 * multiprocessing.cpu_count()
        )
        meta_info = [
            ("ID", str),
            ("x", int),
            ("y", int),
            ("patch_size", int),
            ("annotation", str),
        ] + (
            [(annotation, np.float) for annotation in annotations]
            if not segmentation
            else list([(str(i), np.float) for i in range(target_class)])
        )
        # patch_info = dd.from_delayed(patch_info,meta=meta_info).compute()
        patch_info = patch_info.map_partitions(
            lambda df: df.apply(return_line_info, axis=1), meta=meta_info
        ).compute(
            scheduler="processes"
        )  # .values
        # patch_info=patch_info.apply(return_line_info,axis=1)
        patch_info = patch_info.loc[patch_info["annotation"] != "NA"]
        if segmentation:
            a = 1

    if 0:
        patch_info = dd.from_pandas(
            patch_info, npartitions=2 * multiprocessing.cpu_count()
        )
        meta_info = [
            ("ID", str),
            ("x", int),
            ("y", int),
            ("patch_size", int),
            ("annotation", str),
        ] + (
            [(annotation, np.float) for annotation in annotations]
            if not segmentation
            else list([(str(i), np.float) for i in range(target_class)])
        )
        # patch_info = dd.from_delayed(patch_info,meta=meta_info).compute()
        patch_info = patch_info.map_partitions(
            lambda df: df.apply(return_line_info, axis=1), meta=meta_info
        ).compute(
            scheduler="processes"
        )  # .values
        # patch_info=patch_info.apply(return_line_info,axis=1)
        patch_info = patch_info.loc[patch_info["annotation"] != "NA"]
        if segmentation:
            a = 1
    if 0:
        from parallel_utils import extract_patch_info

        patch_info = extract_patch_info(
            basename,
            input_dir,
            annotations,
            threshold,
            patch_size,
            generate_finetune_segmentation,
            target_class,
            intensity_threshold,
            target_threshold,
        )

    # @pysnooper.snoop("process_line.log")
    def return_line_info(row):
        xs = row["x"]
        ys = row["y"]
        xf = xs + patch_size
        yf = ys + patch_size
        print(basename, xs, ys)
        # if is_valid_patch((purple_mask[xs:xf,ys:yf]>=intensity_threshold).compute(), threshold):#.compute()
        # print(xs,ys, 'valid_patch')
        if segmentation:
            row["annotation"] = "segment"
            # info=[basename,xs,ys,patch_size,'segment']
            seg = segmentation_mask[xs:xf, ys:yf].compute()
            # info=info+
            row.iloc[-target_class:] = [(seg == i).mean() for i in range(target_class)]
            # if generate_finetune_segmentation:
        else:
            row.iloc[-len(annotations) :] = [
                is_coords_in_box(
                    coords=np.array([xs, ys]),
                    patch_size=patch_size,
                    boxes=masks[annotation],
                )
                for annotation in annotations
            ]
            row["annotation"] = annotations[
                row.iloc[-len(annotations) :].argmax()
            ]  # [np.argmax(annotation_areas)]
            # info=[basename,xs,ys,patch_size,main_annotation]+annotation_areas
        # else:
        #     if segmentation:
        #         info = [basename, xs, ys, patch_size, 'NA'] + \
        #             [0. for i in range(target_class)]
        #     else:
        #         info = [basename, xs, ys, patch_size, 'NA'] + \
        #             [0. for i in range(len(annotations))]
        return row  # info

    def seg_line(xs, ys, patch_size, segmentation_mask, target_class):
        xf = xs + patch_size
        yf = ys + patch_size
        seg = segmentation_mask[xs:xf, ys:yf]
        return [(seg == i).mean() for i in range(target_class)]

    def annot_line(xs, ys, patch_size, masks, annotations):
        return [
            is_coords_in_box(
                coords=np.array([xs, ys]),
                patch_size=patch_size,
                boxes=masks[annotation],
            )
            for annotation in annotations
        ]
