import os


def select_path_from_dir(dir, phrase="Select from list: ", suffix="", preselection=None):
    files = [f for f in os.listdir(dir) if (os.path.isfile(os.path.join(dir, f)) and f.endswith(suffix))]

    if preselection is not None and preselection in files:
        file_name = preselection
    else:
        file_name = select_item_from_list(files, phrase=phrase)

    path = os.path.join(dir, file_name)

    return path


def select_item_from_list(input_list, phrase="Select from list: "):
    input_list = sorted(input_list)
    size = len(input_list)

    for idx, item in enumerate(input_list):
        print(f"{idx}: {item}")

    valid_selection = False
    while not valid_selection:
        select_id = input(phrase)

        select_id = int(select_id)
        if select_id in range(size):
            valid_selection = True
        else:
            print("Invalid!")

    selection = input_list[select_id]

    print(selection)

    return selection
