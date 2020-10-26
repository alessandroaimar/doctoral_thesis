from pathlib import Path
main_path = Path(r"D:\DL\datasets\FrameRoshambo")
random_back_path = Path(r"D:\DL\datasets\FrameRoshambo\Random_Background")
filetype = ".jpg"

train_users = [

    "Arthur",
    "David",
    "Helmut",
    "Ivan",
    "Kim",

    "Melvin",


    "Stephan",
    "Jelena",
    "Djordje",
    "Samuele",
    "Ronnie",
    "Filipe",
    "Simi",

]
test_users = [
    "Martin",
    "Michael",
]

val_users = [
    "Pascal",
    "Philipp",
]


symbol_classes = [
    "ROCK", #symbol 0
    "SCISSORS", #symbol 1
    "PAPER", #symbol 2
    "ILLEGAL", #symbol 3
    "BACKGROUND" #symbol 4
]

def save_filelist(filelist, filename, mode, dataset_path):
    filepath = str(dataset_path)+ "\\" + filename
    with open(filepath, mode) as f:
        print("Saving file " + filepath)
        for item in filelist:
            f.write("%s\n" % item)

def generate_image_list(users, filename, dataset_path, root_path, mode):
    filelist = []
    for user in users:

        user_folder = dataset_path / Path(user)

        for class_idx, symbol in enumerate(symbol_classes):
            try:
                symbol_folder = user_folder / (symbol + r"\\DATA\\")
                print(str(symbol_folder))

                for file in symbol_folder.glob("*"+filetype):
                    file = str(file).replace(str(root_path) + "\\", "") #makes paths relative
                    to_filelist_string = str(file) + " " + str(class_idx)
                    print(to_filelist_string)

                    filelist.append(str(to_filelist_string))
            except FileNotFoundError:
                print("Path not found:" + str(symbol_folder))

    save_filelist(filelist, filename,mode, root_path)



#adds the extra used for training datast
def append_extra_training(dataset_path, filename):
    subfolders = ["rock", 'scissors', "paper",]

    extra_path = dataset_path / "rockpaperscissors"
    filelist = list()

    #adding the dataser arthur found online
    for class_idx, subfolder in enumerate(subfolders):

        subfolder_path = extra_path / subfolder

        for file in subfolder_path.glob("*" + ".png"):
            file = str(file).replace(str(dataset_path) + "\\", "")  # makes paths relative
            to_filelist_string = str(file) + " " + str(class_idx)
            print(to_filelist_string)
            filelist.append(str(to_filelist_string))

    save_filelist(filelist, filename,  "a", dataset_path,)



def append_background(dataset_path, filename):
    background_subfolders = ["BACKGROUND_1", "BACKGROUND_2", "BACKGROUND_3"]
    filelist = list()

    for subfolder in background_subfolders:
        subfolder_path = dataset_path / subfolder / "DATA"
        for file in subfolder_path.glob("*.*"):
            file = str(file).replace(str(dataset_path) + "\\", "")  # makes paths relative
            to_filelist_string = str(file) + " " + str(4) #background is 4
            print(to_filelist_string)
            filelist.append(str(to_filelist_string))

    save_filelist(filelist, filename,  "a", dataset_path,)


#images in the main folder
generate_image_list(train_users, "train.txt", main_path, main_path, mode="w+")
#images in the random background folder
generate_image_list(train_users, "train.txt", random_back_path, main_path, mode="a")
append_extra_training(main_path, "train.txt")
append_background(main_path, "train.txt")

generate_image_list(test_users, "test.txt", main_path, main_path, mode="w")
generate_image_list(val_users, "val.txt", main_path, main_path, mode="w")




