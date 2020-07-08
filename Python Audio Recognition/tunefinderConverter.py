import argparse
import warnings
import ConvertWavToMidi
import ConvertMidiToFeatures
import os


def main():
    parser = argparse.ArgumentParser(
        description="Build a Model for testing songs against")

    search = parser.add_argument_group("Parse Parameters")
    # searchhash.add_argument("-hf", "--HashFile", dest="hashfile", type=str,
    # help="Path to the file of hashes to search", required=False)
    search.add_argument("wavPath", type=str, help="Path of wav files")
    search.add_argument("midiPath", type=str, help="Path to store midi files")
    search.add_argument("featurePath", type=str,
                        help="Path to store features")

    search.add_argument("-cl", "--ClipLength", dest="cliplength",
                        type=float, help="Length of clips to create features from", default=8)

    search.add_argument("-fc", "--FeatureCount", dest="featurecount",
                        type=float, help="Number of features to take from an audio clip", default=200)

    # search.add_argument("-s", "--Spread", dest="spread",
    #                     type=float, help="Skip creation of midi from wav", required=False, default=0)

    search.add_argument("-ts", "--TempoSpread", dest="tempospread",
                        type=float, help="How much time should be added/subtracted from the clip length to compensate for variable tempo", required=False, default=0)

    search.add_argument("-sf", "--SkipFeatures", dest="skipfeatures",
                        action="store_true", help="Skip creation of features if features already exist")

    search.add_argument("-sm", "--SkipMidis", dest="skipmidis",
                        action="store_true", help="Skip creation of midi from wav if midi already exists")
    search.add_argument("-ss", "--SkipSpread", dest="skipspread",
                        action="store_true", help="Skip the automatic spread of audio, this is used for creating test features")
    args = vars(parser.parse_args())
    # print(args)
    inputFolder = args["wavPath"]
    midiFolder = args["midiPath"]
    featureFolder = args["featurePath"]
    skipMidi = args["skipmidis"]
    if not os.path.exists(inputFolder):
        print("Audio Folder doesn't exist")
        exit()

    warnings.simplefilter("ignore")
    ConvertWavToMidi.folderWavToMidi(
        inputFolder, midiFolder, skipExistingMidiFiles=skipMidi)

    skipFeatures = args["skipfeatures"]
    clipLength = args["cliplength"]
    featuresPerClip = args["featurecount"]
    if args["skipspread"]:
        spread = 0
    else:
        spread = clipLength-1
    tempoSpread = args["tempospread"]

    ConvertMidiToFeatures.folderMidiToFeatures(
        midiFolder, featureFolder, skipExistingFiles=skipFeatures, secondsPerClip=clipLength, featuresPerClip=featuresPerClip, spread=spread, tempoSpread=tempoSpread)


if __name__ == "__main__":
    main()
