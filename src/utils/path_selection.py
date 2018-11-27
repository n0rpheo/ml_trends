import os


def select_path_from_dir(dir, suffix=None):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    for idx, file in enumerate(files):
        print(f"{idx}: {file}")

    print("geschafft")

    return "ding"