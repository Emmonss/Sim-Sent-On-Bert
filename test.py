

def openfile(filename):
    res= []
    with open(filename,'r',encoding='utf-8') as fr:
        for line in fr:
            res.append(line.strip().split('\t'))
            break
    return res


if __name__ == '__main__':
    filename = './data/paws-x-zh/dev.tsv'
    fieada = openfile(filename)
    print(fieada)
    pass