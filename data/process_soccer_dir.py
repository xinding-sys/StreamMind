import os

def replace_spaces_with_underscores(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        # Rename files
        for file_name in files:
            new_file_name = file_name.replace(" ", "_")
            if new_file_name != file_name:
                os.rename(
                    os.path.join(root, file_name),
                    os.path.join(root, new_file_name)
                )
        
        # Rename directories
        for dir_name in dirs:
            new_dir_name = dir_name.replace(" ", "_")
            if new_dir_name != dir_name:
                os.rename(
                    os.path.join(root, dir_name),
                    os.path.join(root, new_dir_name)
                )

if __name__ == "__main__":
    folder_path = "dataset/MatchTime/dataset/MatchTime/valid"
    replace_spaces_with_underscores(folder_path)

