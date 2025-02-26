import os
from tqdm import tqdm


def add_csv_extension(directory):
    for filename in tqdm(directory,total=len(directory)):
        if filename.endswith(".json") or filename.endswith(".lock"):
            pass
        else:
            new_filename = os.path.join("./", f"{filename}.png")
            old_filepath = os.path.join("./", filename)
            new_filepath = os.path.join("./", new_filename)

            os.rename(old_filepath, new_filepath)
            # print(f"Renamed: {filename} to {new_filename}")

def remove_csv_extension(directory):
    for filename in tqdm(directory,total=len(directory)):
        if filename.endswith(".csv"):
            new_filename = os.path.join("./", f"{filename[:-4]}")
            old_filepath = os.path.join("./", filename)
            new_filepath = os.path.join("./", new_filename)

            os.rename(old_filepath, new_filepath)
            # print(f"Renamed: {filename} to {new_filename}")



import glob 
if __name__ == "__main__":
    directory_path = glob.glob("./dataset/*")  # Replace with the path to your directory
    add_csv_extension(directory_path)
    # remove_csv_extension(directory_path)