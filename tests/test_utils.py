from pathflowai import utils
from numpy import array_equal


# def test_svs2dask_array():
#     from .utils import download_svs
#     from PIL import Image
#     from numpy import array as to_npa
#
#     # from os import remove
#
#     id = "2e4f6316-588b-4629-adf0-7aeac358a0e2"
#     file = "TCGA-MR-A520-01Z-00-DX1.2F323BAC-56C9-4A0C-9C1B-2B4F776056B4.svs"
#     download_location = download_svs(id, file)
#
#     Image.MAX_IMAGE_PIXELS = None  # SECURITY RISK!
#     ground_truth = to_npa(Image.open(download_location))
#
#     test = utils.svs2dask_array(download_location).compute()
#     crop_height, crop_width, _ = test.shape
#
#     # remove(download_location)
#
#     assert array_equal(ground_truth[:crop_height, :crop_width, :], test)


def test_preprocessing_pipeline():
    from .utils import get_tests_dir, image_to_numpy
    from os.path import join, exists

    tests_dir = get_tests_dir()
    basename = "TCGA-18-5592-01Z-00-DX1"
    input_dir = join(tests_dir, "inputs")
    png_file = join(input_dir, basename + ".png")
    xml_file = join(input_dir, basename + ".xml")
    out_zarr = join(tests_dir, "output_zarr.zarr")
    out_pkl = join(tests_dir, "output.pkl")

    # convert a TCGA XML to a binary mask with the following:
    # Image.fromarray(
    #     viewmask.utils.xml_to_image(
    #         ET.parse('./tests/inputs/TCGA-18-5592-01Z-00-DX1.xml')
    #     )
    # ).save('/Users/suman/Downloads/bruh.png')

    utils.run_preprocessing_pipeline(
        png_file, xml_file=xml_file, out_zarr=out_zarr, out_pkl=out_pkl
    )
    assert exists(out_zarr)
    assert exists(out_pkl)

    from zarr import open as open_zarr
    from dask.array import from_zarr as zarr_to_da

    img = zarr_to_da(open_zarr(out_zarr)).compute()
    assert array_equal(img, image_to_numpy(png_file))

    def capture(command):
        from subprocess import Popen, PIPE
        proc = Popen(
            command,
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = proc.communicate()
        return out, err, proc.returncode

    odb = join(tests_dir, "patch_information.db")
    command = [
        "poetry", "run", "pathflowai-preprocess",
        "preprocess-pipeline",
        "-odb", odb,
        "--preprocess",
        "--patches",
        "--basename", basename,
        "--input_dir", input_dir,
        "--patch_size", "256",
        "--intensity_threshold", "45.",
        "-tc", "7",
        "-t", "0.05"
    ]
    out, err, exitcode = capture(command)
    assert exists(out_zarr)
    assert exists(out_pkl)
    assert exists(odb)
    assert exitcode == 0
