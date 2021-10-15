import argparse, os, io

def split_file(file, dir, lines_per_file):
    smallfile = None
    with io.open(file, encoding="utf-8") as bigfile:
        k = 0
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = dir + 'part{0:0=2d}/data/part.txt'.format(k)
                k += 1
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

def make_dirs(dir, k):
    for i in range(k):
        dir_name = dir + "/part{0:0=2d}".format(i)
        os.makedirs(dir_name)
        os.makedirs(dir_name + "/count")
        os.makedirs(dir_name + "/data")
        os.makedirs(dir_name + "/stats")
        os.makedirs(dir_name + "/vocabs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pCTR')
    parser.add_argument('file', help='file to split')
    parser.add_argument('dir', help='directory to save')
    parser.add_argument('fold', help='number of folds to split')
    parser.add_argument('lines', help='number of lines per file')
    args = parser.parse_args()
    make_dirs(args.dir, k = int(args.fold))
    split_file(args.file, args.dir, int(args.lines))
