from pathlib import Path
from Models import *
import cv2

def get_data():
    d_thresh = 20
    img_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #----------------------------------------------
    #Pokemon Image Data Preparation
    images_dir = Path('C:/Users/Graham/Downloads/pokemon_images')
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
    input_data = []
    large_feature_data = []
    micro_feature_data = []
    tiny_feature_data = []
    small_feature_data = []
    medium_feature_data = []

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
                #large_feature_data.append(torch.tensor(np.expand_dims(cv2.resize(poke_map, (img_size, img_size)), axis=0)).float().to(device))
                #medium_feature_data.append(torch.tensor(np.expand_dims(cv2.resize(poke_map, (img_size//2, img_size//2)), axis=0)).float().to(device))
                #small_feature_data.append(torch.tensor(np.expand_dims(cv2.resize(poke_map, (img_size//2//2, img_size//2//2)), axis=0)).float().to(device))
                #tiny_feature_data.append(torch.tensor(np.expand_dims(cv2.resize(poke_map, (img_size//2//2//2, img_size//2//2//2)), axis=0)).float().to(device))
                #micro_feature_data.append(torch.tensor(np.expand_dims(cv2.resize(poke_map, (img_size//2//2//2//2, img_size//2//2//2//2)), axis=0)).float().to(device))
                '''
                poke_map1 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map2 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map3 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map4 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map5 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map6 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map7 = np.zeros((poke_map.shape[0], poke_map.shape[1]))
                poke_map8 = np.zeros((poke_map.shape[0], poke_map.shape[1]))

                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [232, 162, 0]) < d_thresh:
                            poke_map1[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [36, 28, 237]) < d_thresh:
                            poke_map2[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [0, 242, 255]) < d_thresh:
                            poke_map3[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [87, 122, 185]) < d_thresh:
                            poke_map4[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [76, 177, 34]) < d_thresh:
                            poke_map5[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [164, 73, 163]) < d_thresh:
                            poke_map6[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [201, 174, 255]) < d_thresh:
                            poke_map7[x][y] = 1
                for x in range(poke_map.shape[0]):
                    for y in range(poke_map.shape[1]):
                        if np.linalg.norm(poke_map[x][y] - [127, 127, 127]) < d_thresh:
                            poke_map8[x][y] = 1

                poke_mapA = torch.tensor(poke_map1).float()
                poke_mapB = torch.tensor(poke_map2).float()
                poke_mapC = torch.tensor(poke_map3).float()
                poke_mapD = torch.tensor(poke_map4).float()
                poke_mapE = torch.tensor(poke_map5).float()
                poke_mapF = torch.tensor(poke_map6).float()
                poke_mapG = torch.tensor(poke_map7).float()
                poke_mapH = torch.tensor(poke_map8).float()
                poke_floats_large = [poke_mapA, poke_mapB, poke_mapC, poke_mapD, poke_mapE, poke_mapF, poke_mapG, poke_mapH]
                poke_floats_large = torch.stack([x for x in poke_floats_large]).to(device)
                large_feature_data.append(poke_floats_large)

                poke_map1 = cv2.resize(poke_map1, (img_size//2, img_size//2))
                poke_map2 = cv2.resize(poke_map2, (img_size//2, img_size//2))
                poke_map3 = cv2.resize(poke_map3, (img_size//2, img_size//2))
                poke_map4 = cv2.resize(poke_map4, (img_size//2, img_size//2))
                poke_map5 = cv2.resize(poke_map5, (img_size//2, img_size//2))
                poke_map6 = cv2.resize(poke_map6, (img_size//2, img_size//2))
                poke_map7 = cv2.resize(poke_map7, (img_size//2, img_size//2))
                poke_map8 = cv2.resize(poke_map8, (img_size//2, img_size//2))
                poke_mapA = torch.tensor(poke_map1).int().float()
                poke_mapB = torch.tensor(poke_map2).int().float()
                poke_mapC = torch.tensor(poke_map3).int().float()
                poke_mapD = torch.tensor(poke_map4).int().float()
                poke_mapE = torch.tensor(poke_map5).int().float()
                poke_mapF = torch.tensor(poke_map6).int().float()
                poke_mapG = torch.tensor(poke_map7).int().float()
                poke_mapH = torch.tensor(poke_map8).int().float()
                poke_floats_medium = [poke_mapA, poke_mapB, poke_mapC, poke_mapD, poke_mapE, poke_mapF, poke_mapG, poke_mapH]
                poke_floats_medium = torch.stack([x for x in poke_floats_medium]).to(device)
                medium_feature_data.append(poke_floats_medium)

                poke_map1 = cv2.resize(poke_map1, (img_size//2//2, img_size//2//2))
                poke_map2 = cv2.resize(poke_map2, (img_size//2//2, img_size//2//2))
                poke_map3 = cv2.resize(poke_map3, (img_size//2//2, img_size//2//2))
                poke_map4 = cv2.resize(poke_map4, (img_size//2//2, img_size//2//2))
                poke_map5 = cv2.resize(poke_map5, (img_size//2//2, img_size//2//2))
                poke_map6 = cv2.resize(poke_map6, (img_size//2//2, img_size//2//2))
                poke_map7 = cv2.resize(poke_map7, (img_size//2//2, img_size//2//2))
                poke_map8 = cv2.resize(poke_map8, (img_size//2//2, img_size//2//2))
                poke_mapA = torch.tensor(poke_map1).int().float()
                poke_mapB = torch.tensor(poke_map2).int().float()
                poke_mapC = torch.tensor(poke_map3).int().float()
                poke_mapD = torch.tensor(poke_map4).int().float()
                poke_mapE = torch.tensor(poke_map5).int().float()
                poke_mapF = torch.tensor(poke_map6).int().float()
                poke_mapG = torch.tensor(poke_map7).int().float()
                poke_mapH = torch.tensor(poke_map8).int().float()
                poke_floats_small = [poke_mapA, poke_mapB, poke_mapC, poke_mapD, poke_mapE, poke_mapF, poke_mapG, poke_mapH]
                poke_floats_small = torch.stack([x for x in poke_floats_small]).to(device)
                small_feature_data.append(poke_floats_small)

                poke_map1 = cv2.resize(poke_map1, (img_size//2//2//2, img_size//2//2//2))
                poke_map2 = cv2.resize(poke_map2, (img_size//2//2//2, img_size//2//2//2))
                poke_map3 = cv2.resize(poke_map3, (img_size//2//2//2, img_size//2//2//2))
                poke_map4 = cv2.resize(poke_map4, (img_size//2//2//2, img_size//2//2//2))
                poke_map5 = cv2.resize(poke_map5, (img_size//2//2//2, img_size//2//2//2))
                poke_map6 = cv2.resize(poke_map6, (img_size//2//2//2, img_size//2//2//2))
                poke_map7 = cv2.resize(poke_map7, (img_size//2//2//2, img_size//2//2//2))
                poke_map8 = cv2.resize(poke_map8, (img_size//2//2//2, img_size//2//2//2))
                poke_mapA = torch.tensor(poke_map1).int().float()
                poke_mapB = torch.tensor(poke_map2).int().float()
                poke_mapC = torch.tensor(poke_map3).int().float()
                poke_mapD = torch.tensor(poke_map4).int().float()
                poke_mapE = torch.tensor(poke_map5).int().float()
                poke_mapF = torch.tensor(poke_map6).int().float()
                poke_mapG = torch.tensor(poke_map7).int().float()
                poke_mapH = torch.tensor(poke_map8).int().float()
                poke_floats_tiny = [poke_mapA, poke_mapB, poke_mapC, poke_mapD, poke_mapE, poke_mapF, poke_mapG, poke_mapH]
                poke_floats_tiny = torch.stack([x for x in poke_floats_tiny]).to(device)
                tiny_feature_data.append(poke_floats_tiny)

                poke_map1 = cv2.resize(poke_map1, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map2 = cv2.resize(poke_map2, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map3 = cv2.resize(poke_map3, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map4 = cv2.resize(poke_map4, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map5 = cv2.resize(poke_map5, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map6 = cv2.resize(poke_map6, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map7 = cv2.resize(poke_map7, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_map8 = cv2.resize(poke_map8, (img_size//2//2//2//2, img_size//2//2//2//2))
                poke_mapA = torch.tensor(poke_map1).int().float()
                poke_mapB = torch.tensor(poke_map2).int().float()
                poke_mapC = torch.tensor(poke_map3).int().float()
                poke_mapD = torch.tensor(poke_map4).int().float()
                poke_mapE = torch.tensor(poke_map5).int().float()
                poke_mapF = torch.tensor(poke_map6).int().float()
                poke_mapG = torch.tensor(poke_map7).int().float()
                poke_mapH = torch.tensor(poke_map8).int().float()
                poke_floats_micro = [poke_mapA, poke_mapB, poke_mapC, poke_mapD, poke_mapE, poke_mapF, poke_mapG, poke_mapH]
                poke_floats_micro = torch.stack([x for x in poke_floats_micro]).to(device)
                micro_feature_data.append(poke_floats_micro)
                '''
                idx+=1

    '''
    input_data = torch.tensor(input_data)
    input_data = input_data.transpose(2,3).transpose(1,2).float()

    large_feature_data = torch.stack(large_feature_data).float()
    medium_feature_data = torch.stack(medium_feature_data).float()
    small_feature_data = torch.stack(small_feature_data).float()
    tiny_feature_data = torch.stack(tiny_feature_data).float()
    micro_feature_data = torch.stack(micro_feature_data).float()
    

    torch.save(input_data, 'input_data.pt')
    torch.save(large_feature_data, 'large_data.pt')
    torch.save(medium_feature_data, 'medium_data.pt')
    torch.save(small_feature_data, 'small_data.pt')
    torch.save(tiny_feature_data, 'tiny_data.pt')
    torch.save(micro_feature_data, 'micro_data.pt')
    '''
    input_data = torch.load('input_data.pt')
    large_feature_data = torch.load('large_data.pt')
    medium_feature_data = torch.load('medium_data.pt')
    small_feature_data = torch.load('small_data.pt')
    tiny_feature_data = torch.load('tiny_data.pt')
    micro_feature_data = torch.load('micro_data.pt')
    print(input_data.shape[0])
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
    return dataset