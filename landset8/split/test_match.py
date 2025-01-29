import pandas as pd
import re

POSTFIX = ""


def check_duplicate_filenames(csv_file):
    df = pd.read_csv(csv_file, header=None)

    for idx, row in df.iterrows():
        file1, file2 = row[0].strip(), row[1].strip()

        match = re.search(r"(_p\d+\.tif)$", file1)

        if match:
            suffix = match.group(1)
            expected_file2 = file1.replace(suffix, f"{POSTFIX}{suffix}")

            if expected_file2 != file2:
                print(f"Line {idx + 1}: DIFFERENT FILES ({file1}, {file2})")
                return


csv_path = "images_masks.csv"
check_duplicate_filenames(csv_path)
