from pathlib import Path
from Models import *
import cv2

def get_data(train):
    d_thresh = 20
    img_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #----------------------------------------------
    #Pokemon Image Data Preparation
    images_dir = Path('C:/Users/Graham/Downloads/pokemon_images')
    test_maps_dir = Path('C:/Users/Graham/Downloads/test_pokemon_images')
    types_dir = Path('C:/Users/Graham/Downloads/types.txt')
    types_list = open(types_dir, 'r').readlines()
    type_list = torch.zeros(len(types_list), 10, img_size//16, img_size//16)
    for i in range(len(types_list)):
        types = types_list[i].split('\n')[0].split(',')
        for q in types:
            q = int(q.strip('\n')) - 1
            type_list[i][q][:][:] = 1

    real_images = [str(img).replace('\\', '/') for img in images_dir.iterdir() if not "Edit" in str(img)]
    feature_maps = [str(img).replace('\\', '/') for img in images_dir.iterdir() if "Edit" in str(img)]
    test_maps = [str(img).replace('\\', '/') for img in test_maps_dir.iterdir()]
    input_data = []
    large_feature_data = []
    micro_feature_data = []
    tiny_feature_data = []
    small_feature_data = []
    medium_feature_data = []

    if train == False:
        for i in test_maps:
            poke_map = cv2.imread(i, cv2.IMREAD_UNCHANGED)
            poke_map = cv2.cvtColor(poke_map, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(poke_map, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            poke_map[thresh == 255] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            poke_map = cv2.erode(poke_map, kernel, iterations=1)
            poke_map = cv2.resize(poke_map, (img_size, img_size))
            large_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size, img_size))).to(device))
            medium_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size // 2, img_size // 2))).to(device))
            small_feature_data.append(
                torch.tensor(cv2.resize(poke_map, (img_size // 2 // 2, img_size // 2 // 2))).to(device))
            tiny_feature_data.append(
                torch.tensor(cv2.resize(poke_map, (img_size // 2 // 2 // 2, img_size // 2 // 2 // 2))).to(device))
            micro_feature_data.append(
                torch.tensor(cv2.resize(poke_map, (img_size // 2 // 2 // 2 // 2, img_size // 2 // 2 // 2 // 2))).to(device))

            pokemon_image = cv2.imread(i, cv2.IMREAD_UNCHANGED)
            if pokemon_image.shape[2] == 3:
                continue
            trans_mask = pokemon_image[:, :, 3] == 0
            pokemon_image = cv2.cvtColor(pokemon_image, cv2.COLOR_BGRA2BGR)
            pokemon_image[trans_mask] = [0, 0, 0]
            pokemon_image = cv2.resize(pokemon_image, (img_size, img_size))
            input_data.append(pokemon_image)

    if train:
        print("Preprocessing...")
        idx = 0
        for i in real_images:
            for j in feature_maps:
                if i.split("/")[-1].split(".")[0] == j.split("/")[-1].split("Edit")[0]:
                    print(str(idx)+"/"+str(len(real_images)))
                    pokemon_image = cv2.imread(i, cv2.IMREAD_UNCHANGED)
                    poke_map = cv2.imread(j, cv2.IMREAD_UNCHANGED)
                    if pokemon_image.shape[2] == 3:
                        idx+=1
                        continue
                    trans_mask = pokemon_image[:, :, 3] == 0
                    pokemon_image = cv2.cvtColor(pokemon_image, cv2.COLOR_BGRA2BGR)
                    pokemon_image[trans_mask] = [0, 0, 0]
                    poke_map = cv2.cvtColor(poke_map, cv2.COLOR_BGRA2BGR)
                    poke_map[trans_mask] = [0, 0, 0]
                    #poke_map = cv2.cvtColor(poke_map, cv2.COLOR_BGR2GRAY)
                    pokemon_image = cv2.resize(pokemon_image, (img_size, img_size))
                    poke_map = cv2.resize(poke_map, (img_size, img_size))
                    input_data.append(pokemon_image)
                    large_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size, img_size))).to(device))
                    medium_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size//2, img_size//2))).to(device))
                    small_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size//2//2, img_size//2//2))).to(device))
                    tiny_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size//2//2//2, img_size//2//2//2))).to(device))
                    micro_feature_data.append(torch.tensor(cv2.resize(poke_map, (img_size//2//2//2//2, img_size//2//2//2//2))).to(device))
                    idx+=1

    '''
    torch.save(input_data, 'input_data.pt')
    torch.save(large_feature_data, 'large_data.pt')
    torch.save(medium_feature_data, 'medium_data.pt')
    torch.save(small_feature_data, 'small_data.pt')
    torch.save(tiny_feature_data, 'tiny_data.pt')
    torch.save(micro_feature_data, 'micro_data.pt')
    
    input_data = torch.load('input_data.pt')
    #large_feature_data = torch.load('large_data.pt')
    #medium_feature_data = torch.load('medium_data.pt')
    #small_feature_data = torch.load('small_data.pt')
    #tiny_feature_data = torch.load('tiny_data.pt')
    #micro_feature_data = torch.load('micro_data.pt')
    '''
    input_data = torch.tensor(input_data)
    input_data = input_data.transpose(2,3).transpose(1,2).float()
    large_feature_data = torch.stack(large_feature_data)
    medium_feature_data = torch.stack(medium_feature_data)
    small_feature_data = torch.stack(small_feature_data)
    tiny_feature_data = torch.stack(tiny_feature_data)
    micro_feature_data = torch.stack(micro_feature_data)
    large_feature_data = large_feature_data.transpose(2, 3).transpose(1, 2).float()
    medium_feature_data = medium_feature_data.transpose(2, 3).transpose(1, 2).float()
    small_feature_data = small_feature_data.transpose(2, 3).transpose(1, 2).float()
    tiny_feature_data = tiny_feature_data.transpose(2, 3).transpose(1, 2).float()
    micro_feature_data = micro_feature_data.transpose(2, 3).transpose(1, 2).float()
    dataset = [input_data, large_feature_data, medium_feature_data, small_feature_data, tiny_feature_data, micro_feature_data]

    #----------------------------------------------
    #Instantiate dataset with pytorch dataset wrapper class
    class PokemonDataset(Dataset):
        def __init__(self, data):
            self.dataset = data
        def __len__(self):
            return len(self.dataset[0])
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return (self.dataset[0][idx], self.dataset[1][idx], self.dataset[2][idx], self.dataset[3][idx], self.dataset[4][idx], self.dataset[5][idx])

    dataset = PokemonDataset(dataset)
    if train:
        return dataset
    else:
        return [input_data, large_feature_data, medium_feature_data, small_feature_data, tiny_feature_data, micro_feature_data]