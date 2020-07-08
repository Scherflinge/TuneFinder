from PIL import Image, ImageDraw
import ConvertMidiToFeatures
import math
from scipy.stats import zscore

OFF = 0

VALID = 1
DELETE = 2
FUNDAMENTAL = 3
ADD = 4

CONFLICT = 5
CONFLICT_ADD = 6

green_pixel = (0, 255, 0)
red_pixel = (255, 0, 0)
blue_pixel = (0, 0, 255)
black_pixel = (0, 0, 0)
white_pixel = (255, 255, 255)
yellow_pixel = (255, 255, 0)
dark_red_pixel = (144, 0, 0)

ORIGINAL_NOTES = 0
COMPARISON = 1
AFTER = 2

PIXEL_COLORING = {
    (OFF, ORIGINAL_NOTES): black_pixel,
    (OFF, COMPARISON): black_pixel,
    (OFF, AFTER): black_pixel,

    (VALID, ORIGINAL_NOTES): white_pixel,
    (VALID, COMPARISON): white_pixel,
    (VALID, AFTER): white_pixel,

    (FUNDAMENTAL, ORIGINAL_NOTES): white_pixel,
    (FUNDAMENTAL, COMPARISON): green_pixel,
    (FUNDAMENTAL, AFTER): white_pixel,

    (ADD, ORIGINAL_NOTES): black_pixel,
    (ADD, COMPARISON): blue_pixel,
    (ADD, AFTER): white_pixel,

    (DELETE, ORIGINAL_NOTES): white_pixel,
    (DELETE, COMPARISON): red_pixel,
    (DELETE, AFTER): black_pixel,

    (CONFLICT, ORIGINAL_NOTES): white_pixel,
    (CONFLICT, COMPARISON): yellow_pixel,
    (CONFLICT, AFTER): white_pixel,

    (CONFLICT_ADD, ORIGINAL_NOTES): black_pixel,
    (CONFLICT_ADD, COMPARISON): dark_red_pixel,
    (CONFLICT_ADD, AFTER): white_pixel
}


def main():
    try:
        events = ConvertMidiToFeatures.convertMidiToEvents(
            "TuneFinder\\MidtermMidi\\Ain't it fun Test 2.mid")
    except:
        events = ConvertMidiToFeatures.convertMidiToEvents(
            "MidtermMidi\\Ain't it fun Test 2.mid")
    midiend = events[-1][3]

    features_per_second = 20
    width = int(features_per_second*midiend)
    # 127 possible midi notes
    height = 127

    # remove notes if they are too short
    # events = [x for x in events if x[3]-x[2] > 0.1]
    # NOTE This ended up working bad for quick rhythms

    eventgrid = [{} for x in range(width)]
    for note, vel, timeon, timeoff in events:
        beginbucket = int((timeon / midiend)*width)
        endbucket = int((timeoff / midiend)*width)
        for n in range(beginbucket, endbucket):
            eventgrid[n][note] = VALID

    # get rid of found harmonics
    removeHarmonics(eventgrid)

    # get rid of notes outside of standard deviation
    # deleteOutliers(eventgrid)

    # this is for extrapolating values to neighbors
    extrapolateNeighbors(eventgrid)

    for iters in range(3):
        img = Image.new('RGB', (width, height))

        for x in range(width):
            vert = eventgrid[x]
            for y in range(height):
                cur_pixel = vert[y] if y in vert else OFF

                # if cur_pixel == OFF:
                #     value = (0, 0, 0)
                # else:
                #     value = (255, 255, 255)
                # if cur_pixel != OFF:
                #     print(cur_pixel)
                if (cur_pixel, iters) in PIXEL_COLORING:
                    value = PIXEL_COLORING[(cur_pixel, iters)]
                else:
                    value = (0, 0, 0)

                img.putpixel((x, height - y-1), value)
        img.show()
        img.save("midiTest_zscore_{}.png".format(iters))


def removeHarmonics(eventgrid):
    for vert in eventgrid:
        notes = [note for note in vert]
        frequencies = [freq for freq in map(midiNoteToFrequency, notes)]
        frequencies.sort()
        fundamentalFrequency = find_harmonic_fundamental(frequencies)
        fundamentalMidi = None if fundamentalFrequency == None else frequencyToMidiNote(
            fundamentalFrequency)
        # print("{} {}".format(fundamentalMidi, notes))
        if fundamentalMidi:
            for n in notes:
                if n == fundamentalMidi:
                    vert[n] = FUNDAMENTAL
                else:
                    vert[n] = DELETE
            if fundamentalMidi not in notes:
                vert[fundamentalMidi] = ADD
        else:
            for n in notes:
                vert[n] = VALID
        # found_fundamentals.append(vert)


def deleteOutliers(eventgrid, outlierScore=1):
    current_notes = []
    for verts in eventgrid:
        current_notes.extend([x for i, x in enumerate(verts)])

    deviations = list(zscore(current_notes))
    deviated_numbers = [current_notes[i]
                        for i, x in enumerate(deviations) if x > outlierScore]
    deviated_numbers = list(set(deviated_numbers))
    for vert in eventgrid:
        for num in deviated_numbers:
            if num in vert:
                # pass
                vert[num] = DELETE


def extrapolateNeighbors(eventgrid):
    width = len(eventgrid)
    for i, vert in enumerate(eventgrid):
        for note in vert:
            if note in (VALID, OFF, ADD):
                continue
            begin_index = i
            end_index = i
            while(begin_index - 1 >= 0 and note in eventgrid[begin_index-1]):
                begin_index -= 1

            while(end_index + 1 < width and note in eventgrid[end_index+1]):
                end_index += 1

            these_notes = list(set([eventgrid[n][note]
                                    for n in range(begin_index, end_index+1)]))

            val = None

            assert(len(these_notes) > 0)
            if len(these_notes) == 1:
                val = vert[note]
            elif len(these_notes) > 1:
                if DELETE in these_notes and VALID in these_notes:
                    val = DELETE
                elif DELETE in these_notes and ADD in these_notes:
                    val = CONFLICT_ADD
                elif DELETE in these_notes and FUNDAMENTAL in these_notes:
                    val = CONFLICT
                elif VALID in these_notes and FUNDAMENTAL in these_notes:
                    val = FUNDAMENTAL
                elif VALID in these_notes and ADD in these_notes:
                    val = ADD

            for v in range(begin_index, end_index+1):
                if val == None:
                    eventgrid[v][note] = eventgrid[v][note]
                else:
                    eventgrid[v][note] = val


def midiNoteToFrequency(note):
    return 440 * (2**(1./12))**(note-69)
    # 440 * (2^(1/12))^(note-69)


def frequencyToMidiNote(frequency):
    return round((12*math.log(frequency/440))/math.log(2)+69)


def midiNoteToName(note):
    note -= 12
    oct = note//12
    rem = note % 12
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    a = notes[rem]+str(oct)
    return a


def find_harmonic_fundamental(frequencies: list, tolerance=0.05):
    if 0 <= len(frequencies) <= 1:
        return None
    frequencies = frequencies.copy()
    frequencies.sort()
    diffs = []
    for i in range(len(frequencies)-1):
        diffs.append(frequencies[i+1]-frequencies[i])
    # print(diffs)
    iter_range = 1
    if len(frequencies) == 2:
        # iter_range = 1
        pass
    best_found = None
    for iters in range(iter_range):
        if iters == 1:
            # if len(frequencies) == 2:
            #     break

            frequencies.append(frequencies[0]/2)
            frequencies.sort()

        for i, root in enumerate(frequencies):
            ratios = []
            for j, harmonic in enumerate(frequencies):
                # if
                this_ratio = (harmonic/root)
                distance_to_harmonic = this_ratio % 1
                harmonic_ranking = int(this_ratio+0.5)-1
                if distance_to_harmonic > 0.5:
                    distance_to_harmonic = 1-distance_to_harmonic
                harmonic_purity = 1-(distance_to_harmonic*2)
                ratios.append(
                    (distance_to_harmonic, harmonic_ranking, harmonic_purity))
            found_harmonics = [frequencies[i] for i, x in enumerate(ratios) if (
                x[0] < tolerance and x[1] > 0)]
            allscore = [(1/x[1])*x[2] for x in ratios if x[1] > 0]
            score = sum(allscore)

            # if len(found_harmonics) > 1/3*len(ratios):
            if not best_found:
                best_found = (root, score)
            else:
                if score > best_found[1]:
                    best_found = (root, score)
    return best_found[0] if best_found else None


if __name__ == "__main__":
    main()
