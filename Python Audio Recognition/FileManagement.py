import os


def listAllFiles(path, relative=False):
    files = []
    for dirName, folderList, fileList in os.walk(path):
        if len(fileList) > 0:
            if relative:
                files.extend([os.path.relpath(os.path.join(dirName, x), path)
                              for x in fileList])
            else:
                files.extend([os.path.join(dirName, x) for x in fileList])
    return files


def listAllFilesPerDir(path, relative=False):
    files = {}
    for dirName, folderList, fileList in os.walk(path):
        if len(fileList) > 0:
            if relative:
                files[os.path.relpath(dirName, path)] = [os.path.relpath(
                    os.path.join(dirName, x), path) for x in fileList]
            else:
                files[dirName] = [os.path.join(dirName, x) for x in fileList]
    return files


if __name__ == "__main__":
    # path = os.path.join(os.getcwd(), "songs")
    # path = "songs"
    a = listAllFiles("songs", relative=True)
    for x in a:
        print(x)
    print(len(a))
