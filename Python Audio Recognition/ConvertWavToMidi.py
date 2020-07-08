import os
import subprocess
import FileManagement


def __getCommand__(clean=False):
    command = os.path.join("waon-0.11-mingw", "waon.exe")
    shellCommand = command+" -i \"{}\" -o \"{}\" "
    if clean:
        rate = 4096//2
        shellCommand += "-w 6 -n {} -s {}".format(rate, rate//4)
        # shellCommand += "-w 1 -n 4096 -s 1024 -oct 2 -r 2"
        pass

    return shellCommand


def singleWavToMidi(inputPath, outputPath, skipExistingMidiFiles=True):
    if not os.path.exists(inputPath):
        return
    if os.path.exists(outputPath) and skipExistingMidiFiles:
        return
    parentfolder = os.path.dirname(outputPath)
    if not os.path.exists(parentfolder) and parentfolder != "":
        os.makedirs(parentfolder)

    finCommand = __getCommand__(clean=True).format(os.path.join(
        os.getcwd(), inputPath), os.path.join(os.getcwd(), outputPath))
    print("\n"+finCommand+"\n")
    try:
        a = subprocess.check_output(finCommand, shell=True)
    except subprocess.CalledProcessError as e:
        output = "command '{}' return with error (code {}): {}".format(
            e.cmd, e.returncode, e.output.decode())
        print(output)
        exit()


def folderWavToMidi(inputPath, outputPath, skipExistingMidiFiles=True):
    files = FileManagement.listAllFiles(inputPath, relative=True)
    inOuts = [[os.path.join(inputPath, x), os.path.join(outputPath, x)]
              for x in files]

    inOuts = [[x[0], os.path.splitext(x[1])[0]+".mid"] for x in inOuts]

    for x, y in inOuts:
        singleWavToMidi(x, y, skipExistingMidiFiles=skipExistingMidiFiles)


if __name__ == "__main__":
    # folderWavToMidi("midis", "features", skipExistingMidiFiles=False)
    singleWavToMidi(
        "testFe", "test.mid", skipExistingMidiFiles=False)
