def get_tests_dir():
    from os.path import dirname, realpath
    return dirname(realpath(__file__))


def download_svs(id, filename):
    from os import remove
    import requests

    url = f"https://api.gdc.cancer.gov/data/{id}"
    print(f"Downloding from {url}")

    tests_dir = get_tests_dir()
    download_location = f"{tests_dir}/{filename}"

    try:
        remove(download_location)
    except OSError:
        pass

    r = requests.get(url, allow_redirects=True)
    open(download_location, "wb").write(r.content)
    # with requests.get(url, allow_redirects=True, stream=True) as r:
    #     with open(download_location, 'wb') as f:
    #         for chunk in r.iter_content():
    #             if chunk:  # filter out keep-alive new chunks
    #                 f.write(chunk)
    #                 f.flush()

    return download_location
