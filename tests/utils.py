def get_tests_dir():
    from os.path import dirname, realpath

    return dirname(realpath(__file__))


def download_svs(id, filename):
    # from os import remove
    from os.path import join, exists
    import requests

    url = f"https://api.gdc.cancer.gov/data/{id}"
    print(f"Downloding from {url}")

    tests_dir = get_tests_dir()
    download_location = join(tests_dir, filename)

    if exists(download_location):
        # if the file already exists, assume that it's the file we want
        return download_location

    r = requests.get(url, allow_redirects=True)
    open(download_location, "wb").write(r.content)
    # with requests.get(url, allow_redirects=True, stream=True) as r:
    #     with open(download_location, 'wb') as f:
    #         for chunk in r.iter_content():
    #             if chunk:  # filter out keep-alive new chunks
    #                 f.write(chunk)
    #                 f.flush()

    return download_location


def image_to_numpy(image_file):
    # from numpy import array as to_npa
    # from PIL import Image
    # return to_npa(Image.open(image_file))

    from dask_image.imread import imread as img_to_da

    return img_to_da(image_file).compute()
