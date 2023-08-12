import wave
import struct

def backwards(sound):
    """
    Reverses a sound.
    Parameters
    ----------
    sound : A mono sound dictionary with two key/value pairs:
        * "rate": an int representing the sampling rate, samples per second
        * "samples": a list of floats containing the sampled values

    Returns
    -------
    A new mono sound dictionary that is the reverse of sound.

    """
    reversed_samples = []

    #add each sample in backwards order
    for index in range(len(sound["samples"]) - 1, -1, -1):
        reversed_samples.append(sound["samples"][index])
    return {"rate": sound["rate"], "samples": reversed_samples}

def mix(sound1, sound2, p):
    """
    Mixes two sounds together.

    Parameters
    ----------
    sound1 : A mono sound dictionary with two key/value pairs:
        * "rate": an int representing the sampling rate, samples per second
        * "samples": a list of floats containing the sampled values
    sound2 : A mono sound dictionary with two key/value pairs:
        * "rate": an int representing the sampling rate, samples per second
        * "samples": a list of floats containing the sampled values
    p : The mixing parameter

    Returns
    -------
    A new mono sound dictionary, or None if the sampling rates of the two
    sounds are inequal

    """
    if (sound1["rate"]==sound2["rate"]) is False: #if rates not equal
        return None
    rate = sound1["rate"]

    #return new mono sound sample length given samples_key,
    #the key of the sound's samples
    def get_length(samples_key):
        samples1=sound1[samples_key]
        samples2=sound2[samples_key]
        if len(samples1)<=len(samples2):
            return len(samples1)
        return len(samples2)

    #mixes samples1 and samples 2 given length of new mono sound sample
    #and returns the mixed samples
    def mix_samples(length, samples1, samples2):
        mixed_samples = []
        index = 0
        while index <= length:
            altered_sample = p*samples1[index] + samples2[index]*(1 - p)
            mixed_samples.append(altered_sample)
            index += 1
            if index == length:
                break
        return mixed_samples

    if "samples" in sound1.keys():
        length =  get_length("samples")
        mixed_samples = mix_samples(length, sound1["samples"], sound2["samples"])
        return {"rate": rate, "samples": mixed_samples}
    else:
        length =  get_length("left")
        left_mixed_samples = mix_samples(length, sound1["left"], sound2["left"])
        right_mixed_samples = mix_samples(length, sound1["right"], sound2["right"])
        return {"rate": rate, "left": left_mixed_samples, "right": right_mixed_samples}

def convolve(sound, kernel):
    """
    Applies a filter to a sound, resulting in a new sound that is longer than
    the original mono sound by the length of the kernel - 1.
    Does not modify inputs.

    Parameters
    ----------
        sound: A mono sound dictionary with two key/value pairs:
            * "rate": an int representing the sampling rate, samples per second
            * "samples": a list of floats containing the sampled values
        kernel: A list of numbers

    Returns
    -------
        A new mono sound dictionary.
    """
    samples = []  # a list of scaled sample lists

    length = len(sound["samples"])+len(kernel)-1

    # create each scaled sample of same length
    for index, scale in enumerate(kernel):
        if scale != 0:
            scaled_sample = [0] * index
            scaled_sample += [scale * x for x in sound["samples"]]
            scaled_sample += [0] * (length-index-len(sound["samples"]))
            samples.append(scaled_sample)

    # combine samples into one list
    final_sample = [0] * length
    for index in range(length):
        for sample in samples:
            final_sample[index] = final_sample[index] + sample[index]

    return {"rate": sound["rate"], "samples": final_sample}

def echo(sound, num_echoes, delay, scale):
    """
    Compute a new signal consisting of several scaled-down and delayed versions
    of the input sound. Does not modify input sound.

    Parameters
    -------
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled

    Returns
    -------
        A new mono sound dictionary resulting from applying the echo effect.
    """
    sample_delay = round(delay * sound["rate"])
    echo_filter = [1]

    #add scale factors after every sample-delay-th zero
    for index in range(1, num_echoes+1):
        echo_filter += [0] * (sample_delay-1)
        echo_filter += [scale**index]
    return convolve(sound, echo_filter)

def pan(sound):
    """
    Pans audio from the left to right speaker.

    Parameters
    ----------
    sound : sound: A stereo sound dictionary with three key/value pairs:
        * "rate": an int representing the sampling rate, samples per second
        * "left": list of floats with the sampled values for the left speaker
        * "right": list of floats with the sampled values for the right speaker
    Returns
    -------
    A new stereo sound dictionary resulting from applying the pan effect.

    """
    length = len(sound["left"]) #number of samples
    left, right = [sound["left"][0]], [0]
    for index in range(1, length-1): # add scaled samples to left and right
        right += [sound["right"][index]*index/(length-1)]
        left += [sound["left"][index]* (1-index/(length-1))]
    right += [sound["right"][length-1]]
    left += [0]

    return {"rate": sound["rate"], "left": left, "right": right}

def remove_vocals(sound):
    """
    Removes the vocals from a sound.

    Parameters
    ----------
    sound : sound: A stereo sound dictionary with three key/value pairs:
        * "rate": an int representing the sampling rate, samples per second
        * "left": list of floats with the sampled values for the left speaker
        * "right": list of floats with the sampled values for the right speaker
    Returns
    -------
    A new stereo sound dictionary with vocals removed.

    """
    length = len(sound["left"])
    vocalless = []

    # left minus right samples
    for index in range(length):
        vocalless += [sound["left"][index]-sound["right"][index]]
    return {"rate": sound["rate"], "samples": vocalless}



# below are helper functions for converting back-and-forth between WAV files
# and an internal dictionary representation for sounds


def bass_boost_kernel(boost, scale=0):
    """
    Constructs a kernel that acts as a bass-boost filter.

    Start by making a low-pass filter, whose frequency response is given by
    (1/2 + 1/2cos(Omega)) ^ N

    Then scale that piece up and add a copy of the original signal back in.

    Parameters
    -------
        boost: an int that controls the frequencies that are boosted (0 will
            boost all frequencies roughly equally, and larger values allow more
            focus on the lowest frequencies in the input sound).
        scale: a float, default value of 0 means no boosting at all, and larger
            values boost the low-frequency content more);

    Returns
    -------
        A list of floats representing a bass boost kernel.
    """
    # make this a fake "sound" so that one can use the convolve function
    base = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    kernel = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    for i in range(boost):
        kernel = convolve(kernel, base["samples"])
    kernel = kernel["samples"]

    # at this point, the kernel will be acting as a low-pass filter, so
    # scale up the values by the given scale, and add in a value in the middle
    # to get a (delayed) copy of the original
    kernel = [i * scale for i in kernel]
    kernel[len(kernel) // 2] += 1

    return kernel


def load_wav(filename, stereo=False):
    """
    Load a file and return a sound dictionary.

    Parameters
    -------
        filename: string ending in '.wav' representing the sound file
        stereo: bool, by default sound is loaded as mono, if True sound will
            have left and right stereo channels.

    Returns
    -------
        A dictionary representing that sound.
    """
    sound_file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = sound_file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    left = []
    right = []
    for i in range(count):
        frame = sound_file.readframes(1)
        if chan == 2:
            left.append(struct.unpack("<h", frame[:2])[0])
            right.append(struct.unpack("<h", frame[2:])[0])
        else:
            datum = struct.unpack("<h", frame)[0]
            left.append(datum)
            right.append(datum)

    if stereo:
        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = [(ls + rs) / 2 for ls, rs in zip(left, right)]
        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Save sound to filename location in a WAV format.

    Parameters
    -------
        sound: a mono or stereo sound dictionary
        filename: a string ending in .WAV representing the file location to
            save the sound in
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for l_val, r_val in zip(sound["left"], sound["right"]):
            l_val = int(max(-1, min(1, l_val)) * (2**15 - 1))
            r_val = int(max(-1, min(1, r_val)) * (2**15 - 1))
            out.append(l_val)
            out.append(r_val)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)

    # hello = load_wav("sounds/hello.wav")
    # mystery = load_wav("sounds/mystery.wav")
    # write_wav(backwards(hello), "hello_reversed.wav")
    # write_wav(backwards(mystery), "sounds/mystery_reversed.wav")
    water = load_wav("sounds/water.wav", stereo=True)
    synth = load_wav("sounds/synth.wav", stereo=True)
    write_wav(mix(synth, water, 0.3), "sounds/water_synth_mixed_stereo.wav")
    # ice_and_chilli = load_wav("sounds/ice_and_chilli.wav")
    # kernel = bass_boost_kernel(1000, 1.5)
    # write_wav(convolve(ice_and_chilli, kernel), "sounds/bass_boosted.wav")
    # chord = load_wav("sounds/chord.wav")
    # write_wav(echo(chord, 5, 0.3, 0.6), "sounds/chord_echo.wav")
    # car = load_wav("sounds/car.wav", stereo=True)
    # write_wav(pan(car), "sounds/car_pan3.wav")
    # lookout_mountain = load_wav("sounds/lookout_mountain.wav", stereo=True)
    # write_wav(remove_vocals(lookout_mountain), "sounds/lookout_mountain_remove.wav")
