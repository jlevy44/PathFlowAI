from pathflowai import utils
from numpy import array_equal


def test_svs2dask_array():
    from .utils import download_svs
    from PIL import Image
    from numpy import array as to_npa

    # from os import remove

    id = "2e4f6316-588b-4629-adf0-7aeac358a0e2"
    file = "TCGA-MR-A520-01Z-00-DX1.2F323BAC-56C9-4A0C-9C1B-2B4F776056B4.svs"
    download_location = download_svs(id, file)

    Image.MAX_IMAGE_PIXELS = None  # SECURITY RISK!
    ground_truth = to_npa(Image.open(download_location))

    test = utils.svs2dask_array(download_location).compute()
    crop_height, crop_width, _ = test.shape

    # remove(download_location)

    assert array_equal(ground_truth[:crop_height, :crop_width, :], test)


def test_preprocessing_pipeline():
    from .utils import get_tests_dir
    from os.path import join, exists

    tests_dir = get_tests_dir()
    npy_file = join(tests_dir, "inputs/21_5.npy")
    npy_mask = join(tests_dir, "inputs/21_5_mask.npy")
    out_zarr = join(tests_dir, "inputs/21_5.zarr")
    out_pkl = join(tests_dir, "inputs/21_5_mask.pkl")

    utils.run_preprocessing_pipeline(
        npy_file, npy_mask=npy_mask, out_zarr=out_zarr, out_pkl=out_pkl
    )
    assert exists(out_zarr)
    assert exists(out_pkl)

    from zarr import open as open_zarr
    from dask.array import from_zarr as zarr_to_da
    from numpy import load as load_numpy

    img = zarr_to_da(open_zarr(out_zarr)).compute()

    assert array_equal(img, load_numpy(npy_file))

    def capture(command):
        from subprocess import Popen, PIPE
        proc = Popen(
            command,
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = proc.communicate()
        return out, err, proc.returncode

    command = [
        "poetry", "run", "pathflowai-preprocess",
        "preprocess-pipeline",
        "-odb", "patch_information.db",
        "--preprocess",
        "--patches",
        "--basename", "21_5",
        "--input_dir", join(tests_dir, "inputs"),
        "--patch_size", "256",
        "--intensity_threshold", "45.",
        "-tc", "7",
        "-t", "0.05"
    ]
    out, err, exitcode = capture(command)
    assert exitcode == 0
