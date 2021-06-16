import os
import sys
import random
from collections import OrderedDict
from skimage import io
from PIL import Image, ImageOps
from pathlib import Path
from scipy import stats
import numpy as np
import itertools


class Colony:
    N = [-1, 0]
    NE = [-1, 1]
    E = [0, 1]
    SE = [1, 1]
    S = [1, 0]
    SW = [1, -1]
    W = [0, -1]
    NW = [-1, -1]
    directions = [N, NE, E, SE, S, SW, W, NW]

    class Ant:
        def __init__(self, row: int, col: int, colony: "Colony", special=False) -> None:#Cada "formiga" recebe uma posição inicial e se atribui a colônia que pertence"
            self.row = row
            self.col = col
            self.colony = colony
            self.pos_memory = OrderedDict()
            self.special = special

        def __str__(self) -> str:
            """Nicely format an ant to print out"""
            return "[" + str(self.row) + ", " + str(self.col) + "]"

        def probabilistic_choice(self, positions: list, probabilities: list) -> tuple:
            return random.choices(population=positions, weights=probabilities, k=1)[0]
             keys = list(probabilities_to_pos.keys())
             max_key = max(keys)
             return probabilities_to_pos[max_key]
             keys.sort(reverse=True)
             for key in keys:
                 if probabilities_to_pos[key] not in self.colony.pos_memory:
                     return probabilities_to_pos[key]
             return None

        def index_probability(self, index: tuple):#Pega a atribuição feromônio em um índice e o eleva ao controle alpha
            row, col = index #Pega-se a intensidade em um índice para  o controle beta retornando o produto deles
            return (self.colony.pheromone[row, col, -1] ** self.colony.alpha) * \
                   (self.colony.intensities[row, col] ** self.colony.beta)

        def get_index_probabilities(self): #Obtém-se todas as probabilidades de pixels em torna dos valores formiga, retornando o enumerador para a equação
            numerators = list() 
            positions = list()
            random.shuffle(self.colony.directions)
            for d in random.sample(self.colony.directions, len(self.colony.directions)):
                x = self.row + d[0]
                y = self.col + d[1]
                if (x < 0 or x >= self.colony.pheromone.shape[0] or y < 0 or y >= self.colony.pheromone.shape[1]):
                    continue
                if (x, y) in self.colony.pos_memory:
                    continue
                positions.append((x, y))
                numerator = self.index_probability((x, y))
                positions[numerator] = pos
                numerators.append(numerator)
            return numerators, positions #Retorna o enumerado para a equação com a probabilidade das posições correspondentes

        def get_max_probability_pos(self): #Encontra a probabilidade máxima dos quadrados circundantes que não estão na memória
            numerators, positions = self.get_index_probabilities()
            denominator = sum(numerators)
            if (denominator == 0):
                return None
            probabilities = list((x / denominator) * 100 for x in numerators)
            pos_dict = dict(zip(probabilities, positions))
            return self.probabilistic_choice(positions=positions, probabilities=probabilities) #Retorna a posição do quadrado de probabilidade máxima

        def update_memory(self):
            self.colony.pos_memory[(self.row, self.col)] = None
            if (len(self.colony.pos_memory) > self.colony.ant_mem):
                self.colony.pos_memory.popitem(last=False)

        def deposit_pheromone(self): #Deposita o feromônio no local se o limite for excedido, caso contrário, teletransporta a formiga para outro lugar
            self.update_memory()
            pos = self.get_max_probability_pos()
            if (pos == None):
                while (True):
                    pos = (random.randrange(self.colony.img.shape[0]), random.randrange(self.colony.img.shape[1]))
                    if pos not in self.colony.pos_memory:
                        self.row, self.col = pos
                        break
                self.update_memory()
                return
                self.row = random.randrange(self.colony.img.shape[0])
                self.col = random.randrange(self.colony.img.shape[1])
            row, col = pos
            for i, j in np.ndindex(self.colony.pheromone[:2]):
                self.colony.pheromone[i, j, self.colony.memory_index] = 0
            if (self.colony.intensities[row, col] >= self.colony.b):
                self.row = row
                self.col = col
                self.colony.pheromone[row, col, self.colony.memory_index] = self.colony.intensities[row, col]
                self.update_memory()
            else:
                while (True):
                    pos = (random.randrange(self.colony.img.shape[0]), random.randrange(self.colony.img.shape[1]))
                    if pos not in self.colony.pos_memory:
                        self.row, self.col = pos
                        break
            self.update_memory()
                self.row = random.randrange(self.colony.img.shape[0])
                self.col = random.randrange(self.colony.img.shape[1])

    def __init__(self, img_path: str, img: np.ndarray, ant_count=-1, pheromone_evaporation_constant=0.1,
                 pheromone_memory_constant=20, ant_memory_constant=20, minimum_pheromone_constant=0.0001,
                 intensity_threshold_value=-1.0, alpha=1.0, beta=1.0) -> None:
        if (ant_count <= 0):
            ant_count = max(img.shape[0], img.shape[1]) * 3
        self.alpha = alpha
        self.beta = beta
        self.img_path = img_path
        self.img = img
        self.i_max = img.max()
        self.intensities = np.empty(shape=(self.img.shape[0], self.img.shape[1]))
        self.set_pixel_intensities()
        self.generate_intensities_image(invert=False, binary=True)
        self.ants = list()
        #M x N x m + 1 matrix, m + 1 entrada contém feromônio total de outras camadas de memória
        self.pheromone = np.zeros(shape=(img.shape[0], img.shape[1], pheromone_memory_constant + 1))
        self.m = pheromone_memory_constant
        self.pos_memory = OrderedDict()
        self.ant_mem = ant_memory_constant * ant_count
        self.tau_min = minimum_pheromone_constant
        #Inicializa a camada de feromônio total para a quantidade mínima
        for i, j in np.ndindex(self.pheromone.shape[:2]):
            self.pheromone[i, j, -1] = self.tau_min
        # self.pheromone.fill(self.tau_min)
        self.p = pheromone_evaporation_constant
        self.b = intensity_threshold_value if intensity_threshold_value > 0 else self.intensities.mean()
        self.memory_index = 0
        #Define as formigas em todos os pixels aleatórios distintos
        pairs = set()
        first = True
        for ant in range(ant_count):
            pair = None
            while (True):
                #Par não local
                pair = (random.randrange(self.img.shape[0]), random.randrange(self.img.shape[1]))
                if pair not in self.pos_memory:
                    self.pos_memory[pair] = None
                    pairs.add(pair)
                    break
            row, col = pair
            self.ants.append(Colony.Ant(row=row, col=col, colony=self, special=first))
            if (first == True): # (ant % 100 == 0):
                first = False
                self.ants.append(Colony.Ant(row=row, col=col, colony=self, special=first))
            else:
                self.ants.append(Colony.Ant(row=row, col=col, colony=self))

    def __str__(self) -> str:
        ants_list = ['Selection of Ants:\n']
        count = 0;
        for i, ant in enumerate(self.ants):
            if (ant.special == True):
                ants_list.extend(['\t', ant.__str__(), '\n'])
        ants_list.append(self.pheromone[:, :, -1].__str__())
        ants_list.append('\n')
        ants_list.append(self.pheromone[:, :, self.memory_index].__str__())
        return ''.join(ants_list)

    def pixel_intensity(self, row: int, col: int) -> float: #Calcula a intensidade/importância de um determinado pixel olhando para os pixels adjacentes
        (1 / self.i_max) 
        #Parâmetro row - índice x para o pixel
        #Parêmetro col - índice y para o pixel
        return max(
            abs(int(self.img[row - 2, col - 2]) - int(self.img[row + 2, col + 2])) if (row - 2 >= 0 and col - 2 >= 0 and row + 2 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col - 1]) - int(self.img[row + 2, col + 1])) if (row - 2 >= 0 and col - 1 >= 0 and row + 2 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col    ]) - int(self.img[row + 2, col    ])) if (row - 2 >= 0 and row + 2 < self.img.shape[0]) else 0,
            abs(int(self.img[row - 2, col + 1]) - int(self.img[row + 2, col - 1])) if (row - 2 >= 0 and col - 1 >= 0 and row + 2 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col + 2]) - int(self.img[row + 2, col - 2])) if (row - 2 >= 0 and col - 2 >= 0 and row + 2 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col + 2]) - int(self.img[row + 1, col - 2])) if (row - 1 >= 0 and col - 2 >= 0 and row + 1 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row    , col + 2]) - int(self.img[row    , col - 2])) if (col - 2 >= 0 and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row + 1, col + 2]) - int(self.img[row - 1, col - 2])) if (row - 1 >= 0 and col - 2 >= 0 and row + 1 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,

            abs(int(self.img[row - 1, col - 1]) - int(self.img[row + 1, col + 1])) if (row - 1 >= 0 and col - 1 >= 0 and row + 1 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col + 1]) - int(self.img[row + 1, col - 1])) if (row - 1 >= 0 and col - 1 >= 0 and row + 1 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row    , col - 1]) - int(self.img[row    , col + 1])) if (col - 1 >= 0 and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col    ]) - int(self.img[row + 1, col    ])) if (row - 1 >= 0 and row + 1 < self.img.shape[0]) else 0
        )

    def perform_max_normalization_intentsities(self):
        max_val = self.intensities.max()
        for i, j in np.ndindex(self.intensities.shape):
            self.intensities[i, j] /= max_val
        return self.intensities #Retorna intensidade máxima normalizada

    def normalize_intensities(self, zscore=True):
        return stats.zscore(self.intensities) if zscore else self.perform_max_normalization_intentsities()

    def set_pixel_intensities(self): #Cria e armazena todas as intensidade de pixels, sem necessidade de manter a computação, pois o valor nunca muda
        for i, j in np.ndindex(self.img.shape):
            self.intensities[i, j] = self.pixel_intensity(i, j)
        self.intensities = self.normalize_intensities(zscore=False)
        print("Intensity: type: " + str(self.intensities.dtype) + " max: " + str(self.intensities.max()) + " min: " +
              str(self.intensities.min()) + " average intensity: " + str(self.intensities.mean()))
        print(self.intensities)
        Image.fromarray(self.intensities, 'L').show()

    def generate_intensities_image(self, invert=True, binary=True): #Cria e armazena a matriz de intensidades como uma imagem normal em escala de cinza
        base_dir = os.path.dirname(self.img_path)
        intensities_path = os.path.join(base_dir, "Intensities")

        Path.mkdir(intensities_path, parents=True, exist_ok=True)

        base = os.path.basename(self.img_path)
        adjusted_path = os.path.join(os.path.dirname(self.img_path), "intensities_", base)
        final_path = os.path.join(intensities_path, base)
        arr = self.convert_to_gray(self.intensities, binary=binary)
        generate_image_from_array(path=final_path, array=arr, invert=invert)
        Image.fromarray(self.intensities, 'L').save(final_path)

    def adjust_pheromone(self, row, column): #Ajusta globalmente os níveis de feromônios com base nas somas das camadas de memória e os níveis de feromônios antigos
        if (self.memory_index >= self.m - 1):
            self.memory_index = 0
        else:
            self.memory_index += 1
        for i, j in np.ndindex(self.pheromone.shape[:2]):
            print(ij, self.pheromone[ij][!=20])
            deltas = self.pheromone[i, j, :-1]  # The memory layers
            print(stack)
            self.pheromone[i, j, -1] = max((1 - self.p) * self.pheromone[i, j, -1] + sum(deltas), self.tau_min)
            self.pheromone[i, j, self.memory_index] = 0

    def convert_to_gray(self, arr: np.ndarray, binary=True):#Converte uma determinada matriz 2D em escala de cinza
        #Parâmetro arr - Matriz 2D
        arr_fixed = np.zeros(shape=arr.shape, dtype=np.dtype(np.uint8))
        old_max = arr.max()
        old_min = arr.min()
        old_avg = arr.mean()
        print("Converting to image ...")
        print("\tOld type: " + str(arr.dtype) + " Old min: " + str(old_min) + " old max: " + str(old_max) +
              " old range: " + str(old_max - old_min) + " old average: " + str(old_avg))
        for i, j in np.ndindex(arr.shape):
            if (binary == True):
                if (arr[i, j] >= old_avg):
                    arr_fixed[i, j] = 255
                else:
                    arr_fixed[i, j] = 0
            else:
                arr_fixed[i, j] = int((((255 - 0) * (arr[i, j] - old_min)) / (old_max - old_min)) + 0 + 0.5)
        print("\tNew type: " + str(arr_fixed.dtype) + " new min: " + str(arr_fixed.min()) + " new max: " +
              str(arr_fixed.max()) + " new range: " + str(arr_fixed.max() - arr_fixed.min()) + " new average: " +
              str(arr_fixed.mean()))
        return arr.astype(dtype=np.dtype(np.uint8), casting='unsafe', copy=True)

    def generate_pheromone_image(self, iteration): #Gera a camada de feromônio atual com o número de iteção dado para o nome
        #Paramêtro iteration integer - Vai nomear o arquivo
        base_dir = os.path.dirname(self.img_path)
        base = os.path.basename(self.img_path)
        fname = base.split('.')[0]

        iterations_path = os.path.join(base_dir, "Iterations", fname)

        final_path = os.path.join(iterations_path, str(iteration) + "_" + base)
        arr = self.convert_to_gray(self.pheromone[:, :, -1])
        old_max = arr.max()
        old_min = arr.min()
        for i, j in np.ndindex(arr.shape):
            arr[i, j] = ((255 * (arr[i, j] - old_min)) / (old_max - old_min))
        generate_image_from_array(path=final_path, array=arr)

    def clean_up(self, dir_path): #Limpa o diretório dir_path
        #Paramêtro dir_path - diretório para limpar
        print("cleaning " + dir_path + " ...")
        entries = None
        try:
            entries = os.listdir(path=dir_path)
        except FileNotFoundError as FNFE:
            print(type(FNFE).__name__ + ": " + argv[0] + ": " + str(FNFE), file=sys.stderr)
            print("Nothing to clean!")
            return
        for entry in entries:
            path = os.path.join(dir_path, entry)
            try:
                os.remove(path=path)
            except FileNotFoundError:
                break
        print("Done cleaning!")

    def iterate(self, iterations=-1): #Executa iteções, número de iterações do algoritmo ACO
        #Paramêtro iterations: número de iterações para performance
        if iterations <= 0:
            iterations = max(self.img.shape[0], self.img.shape[1])
        print("Iterations: " + str(iterations))
        for i in range(iterations):
            print("Iteration: " + str(i + 1))
            if ((i + 1) % 10 == 0):
                print("Iteration: " + str(i + 1))
                self.generate_pheromone_image(iteration=(i + 1))
            for ant in self.ants:
                ant.deposit_pheromone()
            if (True): ((i + 1) % 10 == 0):
                print(self)
            self.adjust_pheromone()
        print("Max: " + str(self.pheromone[:, :, -1].max()))
        print(self.pheromone[:, :, -1])
        print(self.convert_to_gray(self.pheromone[:, :, -1]))


def generate_image_from_array (path, array, invert=True): #Inverte o esquema do array e o salva fisicamente
    #Paramêtro path - diretório para armazenamento
    #Paramêtro array - 2D array de dados
    dir_base = os.path.dirname(path)
    Path(dir_base).mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(array, 'L')
    if (invert == True):
        img = ImageOps.invert(img)
    img.save(path)


if __name__ == "__main__":
    argv = sys.argv
    print("Args: " + str(argv))
    if (len(argv) != 2):
        print("Usage: " + argv[0] + ": `" + os.path.basename(argv[0]) + " <image directory>'", file=sys.stderr)
        exit(1)

    entries = None
    try:
        entries = os.listdir(path=argv[1])
    except FileNotFoundError as FNFE:
        print(type(FNFE).__name__ + ": " + argv[0] + ": " + str(FNFE), file=sys.stderr)
        exit(2)

    print("Directory [" + argv[1] + "]: " + str(entries))

    for item in entries:
        if (item == "Intensities" or item == "Iterations"):
            continue
        path = os.path.join(argv[1], item)
        try:
            img = io.imread(path)
            print("Read in: `" + path + "'")
        except ValueError as VE:
            print(type(VE).__name__ + ": " + argv[0] + ": `" + path + "' is not a valid image", file=sys.stderr)
            continue
        print("[" + item + "] size: " + str(img.shape) + " len: " + str(img.shape[0] * img.shape[1]))
        print(img)
        print("Image: type: " + str(img.dtype) + " max: " + str(img.max()) + " min: " + str(img.min()) + " mean: " +
              str(img.mean()))
        c = Colony(img_path=path, img=img, ant_count=-1, pheromone_evaporation_constant=0.001,
                   pheromone_memory_constant=30, ant_memory_constant=30, intensity_threshold_value=-1.0,
                   alpha=2.5, beta=2.0)
        clean_path = os.path.join(argv[1], "Iterations", item.split('.')[0])
        c.clean_up(dir_path=clean_path)
        c.iterate(iterations=-1)
        c.adjust_pheromone()
