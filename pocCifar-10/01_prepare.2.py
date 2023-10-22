import os
import shutil
import random
from tqdm import tqdm

#ENTER THE NAME OF TARGET DIRECTORY:

TARGET_DIR = "data"


def transform_data_structure(source_dir, source_dir_TT, target_dir):
    # Create target directories if they don't exist
    
    os.makedirs(target_dir, exist_ok=True)

    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'testFull'), exist_ok=True)

    class_folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]

    class_folders_TT = [folder for folder in os.listdir(source_dir_TT) if os.path.isdir(os.path.join(source_dir_TT, folder))]

    for class_folder in class_folders:

        class_path = os.path.join(source_dir, class_folder)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images = [img for img in images if not img.startswith("._")] #remove ._ files

        num_images = len(images)

        random.shuffle(images)  # Shuffle the list of images

        train_split = int(0.6 * num_images)
        val_split = int(0.2 * num_images)
        test_split = int(0.2 * num_images)  # Smaller test set

        all_images = images[:train_split + val_split + test_split]

        random.shuffle(all_images)  # Shuffle the images

        train_images = all_images[:train_split]
        val_images = all_images[train_split:train_split + val_split]
        test_images = all_images[train_split + val_split:]
        
        class_target_dir = os.path.join(target_dir, 'train', class_folder)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for train split for {class_folder} subset")
        for img in tqdm(train_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)

        # class_target_dir = os.path.join(target_dir, 'val', class_folder)
        # os.makedirs(class_target_dir, exist_ok=True)
        # print(f"preparing data for val split for {class_folder} subset")
        # for img in tqdm(val_images):
        #     shutil.copy2(os.path.join(class_path, img), class_target_dir)

        class_target_dir = os.path.join(target_dir, 'testFull', class_folder)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for test split for {class_folder} subset")
        for img in tqdm(test_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)

    for class_folder_TT in class_folders_TT:

        class_path = os.path.join(source_dir_TT, class_folder_TT)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images = [img for img in images if not img.startswith("._")] #remove ._ files

        num_images = len(images)

        random.shuffle(images)  # Shuffle the list of images

        train_split = int(0.6 * num_images)
        val_split = int(0.2 * num_images)
        test_split = int(0.2 * num_images)  # Smaller test set

        all_images = images[:train_split + val_split + test_split]

        random.shuffle(all_images)  # Shuffle the images

        train_images = all_images[:train_split]
        val_images = all_images[train_split:train_split + val_split]
        test_images = all_images[train_split + val_split:]
        
        class_target_dir = os.path.join(target_dir, 'train', class_folder_TT)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for train split for {class_folder_TT} subset")
        for img in tqdm(train_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)

        class_target_dir = os.path.join(target_dir, 'val', class_folder_TT)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for val split for {class_folder_TT} subset")
        for img in tqdm(val_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)

        class_target_dir = os.path.join(target_dir, 'test', class_folder_TT)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for test split for {class_folder_TT} subset")
        for img in tqdm(test_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)

        class_target_dir = os.path.join(target_dir, 'testFull', class_folder_TT)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for test split for {class_folder_TT} subset")
        for img in tqdm(test_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)


if __name__ == "__main__":
    
    random.seed(2137)
    
    source_directory = "/home/macierz/s175856/TransferTesting/pocCifar-10/Transfer_testing_db/TT_DB"  # Update this to your source directory
    source_directory_TT = "/home/macierz/s175856/TransferTesting/pocCifar-10/Transfer_testing_db/AddClass"  # Update this to your source directory
    target_directory = f"/home/macierz/s175856/TransferTesting/pocCifar-10/Transfer_testing_db/{TARGET_DIR}"  # Update this to your desired target directory

    transform_data_structure(source_directory, source_directory_TT, target_directory)

