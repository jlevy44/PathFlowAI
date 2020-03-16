def test_svs2dask_array():
    from pathflowai import utils
    from .utils import download_svs
    from PIL import Image
    import numpy as np
    from os import remove

    id = "2e4f6316-588b-4629-adf0-7aeac358a0e2"
    file = "TCGA-MR-A520-01Z-00-DX1.2F323BAC-56C9-4A0C-9C1B-2B4F776056B4.svs"
    download_location = download_svs(id, file)

    Image.MAX_IMAGE_PIXELS = None  # SECURITY RISK!
    ground_truth = np.array(Image.open(download_location))

    test = utils.svs2dask_array(download_location).compute()
    crop_height, crop_width, _ = test.shape

    remove(download_location)

    assert np.array_equal(ground_truth[:crop_height, :crop_width, :], test)
