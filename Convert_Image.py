from PIL import Image, UnidentifiedImageError
import sys
import os

if __name__ == "__main__":
    argv = sys.argv
    print("Args: " + str(argv))
    if (len(argv) < 2 or len(argv) > 3):
        print("Usage: " + argv[0] + ": `" + os.path.basename(argv[0]) + " <directory from> [<directory to>]'", #Seleção do diretório das imagens originais
              file=sys.stderr)
        exit(1) #Saída do processamento da imagem
        
    entries = None
    try:
        entries = os.listdir(path=argv[1])
    except FileNotFoundError as FNFE:
        print(type(FNFE).__name__ + ": " + argv[0] + ": " + str(FNFE))
        exit(2)

    print("Directory [" + argv[1] + "]: " + str(entries))    

    for item in entries: #Entrada da escala em cinza para mapeamento das imagens originais
        error = False
        fin = None
        if (item == "Grey-Scale"):
            continue
        try:
            filename = os.path.join(argv[1], item)
            fin = Image.open(filename)
        except PermissionError as PE:
            print(type(PE).__name__ + ": " + argv[0] + ": item `" + filename + "' is not a file") #Saída de inserção do arquivo, caso ela não seja um arquivo de formato
            continue
        except UnidentifiedImageError as UIE:
            print(type(UIE).__name__ + ": " + argv[0] + ": item `" + filename + "' is not an image") #Saída da inserção da imagem, caso ela não seja uma imagem
            continue
        fin = fin.convert('L')
        if (len(argv) == 3):
            try:
                filename = os.path.join(argv[2], item)
                fin.save(filename) #Extensão para salvar a(s) imagem(s) processadas, apontamento de pasta
                print("Converted and saved [" + item + "] to directory [" + argv[2] + "]") #Printando saída de resultado de conversão
            except FileNotFoundError as FNFE:
                print(type(FNFE).__name__ + ": " + argv[0] + ": invalid path `" + filename +
                      "'\n\t- will attempt to save within [" + argv[1] + "]")
                error = True

        if(len(argv) == 2 or error == True): #Loop para verificar existência do arquivo
            dir_path = os.path.join(argv[1], "Gray-Scale")
            path = None
            try:
                os.mkdir(dir_path)
            except FileExistsError:
                pass
            try:
                path = os.path.join(dir_path, item)
                fin.save(path)
            except FileNotFoundError as FNFE:
                print(type(FNFE).__name__ + ": " + argv[0] + ": invalid path `" + path + "'")
                exit(2)
            print("Converted and saved [" + item + "] to directory [" + dir_path + "]") #Saída resultado do loop

    print("Done")
