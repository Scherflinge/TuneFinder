import mido
import time
from sklearn import preprocessing
import numpy as np
import math
import os
import json
import FileManagement
import shutil
from itertools import product
import progressbar
import heapq


def convertMidiToEvents(midifile):
    '''
    returns ['note', 'velocity', 'time on', 'time off']s
    '''
    mid = mido.MidiFile(midifile)
    events = []
    for track in mid.tracks:
        tick = 0
        currentTime = 0
        holdoverNotes = {}
        for msg in track:
            if msg.__class__.__name__ is not "MetaMessage":
                # Update the summed time
                tick += msg.time
                currentTime = mido.tick2second(
                    tick, mid.ticks_per_beat, 500000)

                if msg.type == 'note_on':
                    holdoverNotes[msg.note] = [
                        msg.note, msg.velocity, currentTime]
                elif msg.type == 'note_off':
                    # TODO maybe it should check if we're in a new feature first?
                    #
                    # Once a note is released, add it to to the current feature
                    info = holdoverNotes.pop(msg.note)
                    info.append(currentTime)
                    events.append(info)
        if len(holdoverNotes.keys()) > 0:
            raise Exception("Notes didn't end")
    return events


def convertEventsToMidi(events):
    midievents = []
    for note, velocity, timeon, timeoff in events:
        heapq.heappush(midievents, (timeon, "on", note, velocity))
        heapq.heappush(midievents, (timeoff, "off", note, velocity))
    midievents = [heapq.heappop(midievents) for x in range(len(midievents))]

    mid = mido.MidiFile()
    t = mid.add_track(name="Test Track")
    lastTick = 0
    for ev in events:
        tick = mido.second2tick(ev[0], mid.ticks_per_beat, 500000)
        tickDif = tick - lastTick
        lastTick = tick
        mType = "note_on" if ev[2] == "on" else "note_off"
        tickDif += 0.5
        tickDif = math.floor(tickDif)
        t.append(mido.Message(mType, note=ev[1], time=tickDif))

    t.append(mido.MetaMessage("end_of_track"))
    return midievents


def timetoindex(time, cliplength, featurecount, offset):
    t = time+offset
    clip = 0
    clip = t // cliplength
    if offset < 0:
        clip += abs(offset // cliplength)

    feature = (t % cliplength) // (cliplength/featurecount)
    return int(feature + clip*featurecount)


def midiToFeatures(events, offset=0, secondsPerClip=20, featuresPerClip=40):
    # mid = mido.MidiFile(fileName)

    # fullTime = mid.length
    numFeatures = featuresPerClip

    secondsPerFeature = secondsPerClip/featuresPerClip
    features = []
    for note, velocity, timeon, timeoff in events:
        pass
        start = timetoindex(timeon, secondsPerClip, featuresPerClip, offset)
        end = timetoindex(timeoff, secondsPerClip, featuresPerClip, offset)
        while len(features) < end:
            features.append([])
        for i in range(start, end, 1):
            features[i].append((note, velocity))
    while(len(features) % featuresPerClip != 0):
        features.append([])
    lengths = list(map(len, features))
    avgfeatures = list(map(notesAverage, features))
    clips = []
    for i in range(int(len(avgfeatures)//featuresPerClip)):
        normalized = normalizeData(
            avgfeatures[int(i*numFeatures):int((i+1)*numFeatures)])
        if normalized != None:
            clips.append(normalized)

    return clips


def notesAverage(notes):
    # If no notes are given, respond "NA"
    if notes == [] or notes == {}:
        return np.nan
    # Description:
    # Average notes using velocity
    noteVals = None
    weights = None
    if isinstance(notes, dict):
        noteVals = [notes[x][0] for x in notes.keys()]
        weights = [notes[x][1] for x in notes.keys()]
    else:
        noteVals = [x[0] for x in notes]
        weights = [x[1] for x in notes]
    for g in range(len(noteVals)):
        noteVals[g] = noteVals[g] * weights[g] / sum(weights)
    return round(sum(noteVals), 3)


def normalizeData(data):
    num = []
    nas = []
    for i in range(len(data)):
        # a = type(data[i])
        if math.isnan(data[i]):
            nas.append((i, 0))
        else:
            num.append(data[i])

    if len(num) < (1/3 * len(data)):
        return None

    mi = min(num)
    num2 = [x-mi for x in num]
    ma = max(num2)
    if ma == 0:
        num2 = [15 for x in num]
    else:
        num2 = [(x/ma)*10+10 for x in num2]
        # try:
        #     num = preprocessing.normalize([num]).tolist()[0]
        # except Exception as w:
        #     print(w)
        #     exit()
    for x in nas:
        num2.insert(x[0], x[1])
    return num2


def fileMidiToFeatures(inputFile, outputFile, offset=0, secondsPerClip=20, featuresPerClip=40, skipExistingFiles=False, spread=0, tempoSpread=0):
    if not os.path.exists(inputFile):
        return

    if not os.path.exists(os.path.dirname(outputFile)):
        if os.path.dirname(outputFile) is not "":
            os.makedirs(os.path.dirname(outputFile))
    elif os.path.exists(outputFile):
        if skipExistingFiles:
            return

    print("Converting {} to features, {}...".format(inputFile, outputFile))
    result = []
    spRange = [0]
    if spread != 0:
        # spRange = range(-spread, spread)
        spRange = np.linspace(-spread, 0, spread*2+1).tolist()
    tmpRange = [0]
    if tempoSpread != 0:
        tmpRange = np.linspace(
            secondsPerClip-tempoSpread, secondsPerClip+tempoSpread, (tempoSpread*2)*2+1).tolist()

    combos = [x for x in product(spRange, tmpRange)]
    counter = 0
    events = convertMidiToEvents(inputFile)
    for i, j in progressbar.progressbar(combos):
        result.extend(midiToFeatures(events, offset=offset+i,
                                     secondsPerClip=secondsPerClip+j, featuresPerClip=featuresPerClip))

    writeData(outputFile, result)

    print("Finished conversion.\n")


def writeData(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data))


def folderMidiToFeatures(inputFolder, outputFolder, offset=0, secondsPerClip=20, featuresPerClip=40, skipExistingFiles=False, spread=0, tempoSpread=0):
    print("Converting folder {} to features, {}...".format(
        inputFolder, outputFolder))
    for x in FileManagement.listAllFiles(inputFolder, relative=True):
        fileMidiToFeatures(os.path.join(inputFolder, x),
                           os.path.join(outputFolder, os.path.splitext(x)[0]+".json"), offset=offset, secondsPerClip=secondsPerClip, featuresPerClip=featuresPerClip, skipExistingFiles=skipExistingFiles, spread=spread, tempoSpread=tempoSpread)
    print("Finished Folder Conversion.")


if __name__ == "__main__":
    a = convertMidiToEvents("new Midi Test.mid")
    b = convertEventsToMidi(a)
    exit()
    # fileMidiToFeatures("test.mid", "testfeature.json",
    #    secondsPerClip=8, featuresPerClip=200)
