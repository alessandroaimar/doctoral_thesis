log_path = r"D:\DL\models\SparseRoshambo\baseline\run_log.txt"

fileh = open(log_path, 'r')


def checkForTensor(line):
    if "Tensor(" in line:

        num_slash = line.count(r"/")
        if num_slash == 1:
            start_char = '"'
        else:
            start_char = r"/"

        start = line.find(start_char)
        start_cleaned = line[start + 1:]
        end = start_cleaned.find(r'/')

        return start_cleaned[:end]
    else:
        return None


def checkForFloat(line):
    try:
        return float(line)
    except ValueError:
        return None


def isSparsity(tensorName):
    if "sparsity" in tensorName:
        return True
    else:
        return False


def isExponent(tensorName):
    if "exp" in tensorName:
        return True
    else:
        return False


def isEpoch(tensorName):
    if "Epoch" in tensorName:
        return True
    else:
        return False

def isActiv(tensorName):
    if "activ" in tensorName or "layer_output" in tensorName:
        return True
    else:
        return False

def isKernel(tensorName):
    if "kernel" in tensorName:
        return True
    else:
        return False


def getEpoch(cleanLine):
    epoch_start_index = cleanLine.find("Epoch")
    noEpochLine = cleanLine[epoch_start_index + 6:]
    epochIdxEnd = noEpochLine.find(" ")
    epochString = noEpochLine[0:epochIdxEnd]

    return int(epochString)


def preprocessLine(line):
    if "]" in line:
        split = line.split("] ")
        return split[1]
    else:
        return line


sparsities_activ = dict()
exponents_activ = dict()

sparsities_kern = dict()
exponents_kern = dict()

for line in fileh.readlines():

    clean_line = preprocessLine(line)

    tmpName = checkForTensor(clean_line)
    tmpFloat = checkForFloat(clean_line)

    if isEpoch(clean_line):
        epoch = getEpoch(clean_line)

    elif tmpName is not None:
        # tensor name found, so no integer/float found
        tensorName = tmpName
    elif tmpFloat is not None:
        append_dict = {"epoch": epoch,
                       "value": tmpFloat}


        if isActiv(tensorName):
            sparsities = sparsities_activ
            exponents = exponents_activ
        elif isKernel(tensorName):
            sparsities = sparsities_kern
            exponents = exponents_kern
        else:
            print("Error?")


        if isSparsity(tensorName):
            try:
                print("Appending sparsity value {} to tensor {}".format(tmpFloat, tensorName))
                sparsities[tensorName].append(append_dict)
            except KeyError:
                print("Creating sparsity list {}".format(tensorName))
                sparsities[tensorName] = [append_dict]
        elif isExponent(tensorName):
            try:
                print("Appending exponent value {} to tensor {}".format(tmpFloat, tensorName))
                exponents[tensorName].append(append_dict)
            except KeyError:
                print("Creating exponent list {}".format(tensorName))
                exponents[tensorName] = [append_dict]
        else:
            print("ERROR? {}".format(tensorName))



import matplotlib.pyplot as plt

figure, axis = plt.subplots(nrows=2, ncols=2)


def add_subplot(dictionary, label, position, filter=[]):

    for tensor, all_items in dictionary.items():
        skip = False
        for filter_value in filter:
            if filter_value in tensor:
                skip=True

        if not skip:
            x = []
            y = []

            for sparsity in all_items:
                x.append(sparsity["epoch"])
                y.append(sparsity["value"])
            axis[position].set_title(label)
            line, = axis[position].plot(x, y)
            line.set_label(tensor)

    axis[position].legend()

#ACTIV
# sparsity subplot
add_subplot(sparsities_activ, "Activations Sparsity", (0,0), ["layer6"])
add_subplot(exponents_activ, "Activations Exponents", (0,1))

add_subplot(sparsities_kern, "Kernels Sparsity", (1,0))
add_subplot(exponents_kern, "Kernels Exponent", (1,1))


plt.show()
