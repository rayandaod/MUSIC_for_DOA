import numpy as np
import scipy.signal as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.io import wavfile
from scipy import signal

SPEED_OF_SOUND = 343  # m/s
data_path = Path("../data")


def mic_distances(mic_locations):
    d = []
    for l in mic_locations:
        d.append(np.linalg.norm(l-mic_locations[0]))
    return d


def load_and_plot(file_shortname, filenames, retrieve_angle=False):
    chosen_file_path = data_path / filenames[file_shortname]

    if retrieve_angle:
        directions_of_arrival = [int(file_shortname[-2:])]
    else:
        directions_of_arrival = None

    # Load the audio file
    fs, data = wavfile.read(data_path / chosen_file_path)

    # Normalisation
    data = data / np.max(np.array(data), axis=0)

    # Plot the waveform
    plt.plot(np.arange(len(data[:, 0])) / fs, data[:, 0])
    plt.title("Loaded audio file")
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.show()

    return fs, data, directions_of_arrival, chosen_file_path


# Find the angles given the Spectrum Function
def find_angles(P_MUSIC, peak_search_range, angles_to_be_found, prominence=0.1, width=1):
    peaks, _ = sc.find_peaks(P_MUSIC, prominence=prominence, width=width)
    peaks = peaks * abs(peak_search_range[2] - peak_search_range[1]) + peak_search_range[0]
    print("Actual angle(s) is/are:", angles_to_be_found)
    print("Angle(s) found is/are:", peaks)
    print(np.sum(np.sort(np.array(peaks)) == np.sort(np.array(angles_to_be_found))), "/", len(peaks),
          "correct guesses")
    return peaks


def time_frames(audio_len, N=1024):
    hop_length = N / 2
    index = 0
    time_frame_indices = [0]
    while index + hop_length < audio_len:
        index += hop_length
        time_frame_indices.append(int(index))
    return time_frame_indices


def find_n_freqs(data_focus, fs, n_freqs, prominence, freq_range=None):
    # Compute the Power Spectral Density
    data_focus_centered = data_focus - np.mean(data_focus)
    Xw = np.fft.fft(data_focus_centered)
    L = np.size(Xw) / 2
    freq_indices = np.arange(0, fs / 2, fs / 2 / L)
    psd = np.abs(Xw[:int(L)])
    psd = psd / max(psd)

    # Find the required number of peaks in the PSD
    if freq_range is None:
        peak_indices = sc.find_peaks(psd, prominence=prominence)[0]
        peak_values = psd[peak_indices]
        n_largest_peak_indices_in_peak_values = np.argsort(peak_values)[::-1][:n_freqs]
        n_peak_freqs = freq_indices[peak_indices[n_largest_peak_indices_in_peak_values]]
    else:
        start_freq = int(np.floor(freq_range[0]*2*L/fs))
        end_freq = int(np.floor(freq_range[1]*2*L/fs))
        start_to_end = np.arange(start_freq, end_freq)
        peak_indices = sc.find_peaks(psd[start_to_end], prominence=prominence)[0]
        peak_values = psd[start_to_end][peak_indices]
        n_largest_peak_indices_in_peak_values = np.argsort(peak_values)[::-1][:n_freqs]
        n_peak_freqs = freq_indices[start_to_end][peak_indices[n_largest_peak_indices_in_peak_values]]

    return psd, freq_indices, n_peak_freqs


def frame_is_too_quiet(audio_frame, threshold):
    max_value = max(audio_frame)
    assert max_value <= 1, "The audio frame is not normalised!"
    if max_value <= threshold:
        return True


def get_freqs_by_range(audio, fs, mic_indices, n_freqs=1, audio_time_range=None, N=1024, prominence=0.5,
                       plot=False, quiet_threshold=0.3, freq_range=None):
    # Make sure that the audio file is bigger than the frame length
    audio_len = audio.shape[0]
    assert audio_len >= N, "The audio file is smaller than the frame_length!"

    # Make sure the audio is normalised wrt its maximum value
    audio = audio / np.max(np.array(audio), axis=0)

    # Initialise return variables
    X = []
    freqs_by_time_range = dict()

    # If audio_time_range is None, we have to find the main frequencies in all the frames
    if audio_time_range is None:
        time_frame_indices = time_frames(audio_len, N)
        n_time_frame_indices = len(time_frame_indices)

        index = 0
        while index + 2 < n_time_frame_indices:
            audio_time_range = np.arange(time_frame_indices[index], time_frame_indices[index + 2])
            X.append(audio[audio_time_range, :][:, mic_indices].T)

            data_focus = audio[audio_time_range, 0]
            if frame_is_too_quiet(data_focus, threshold=quiet_threshold):
                freqs_by_time_range[f"{audio_time_range[0]}, {audio_time_range[-1]}"] = None
            else:
                psd, freq_indices, main_freqs = find_n_freqs(data_focus, fs, n_freqs, prominence, freq_range)
                freqs_by_time_range[f"{audio_time_range[0]}, {audio_time_range[-1]}"] = main_freqs

            index += 1

    # Else, we only find the main frequencies in the given time range
    else:
        X.append(audio[audio_time_range, :][:, mic_indices].T)

        data_focus = audio[audio_time_range, 0]
        psd, freq_indices, main_freqs = find_n_freqs(data_focus, fs, n_freqs, prominence, freq_range)
        freqs_by_time_range[f"{audio_time_range[0]}, {audio_time_range[-1]}"] = main_freqs

    if plot:
        for key, value in freqs_by_time_range.items():
            string_list = key.split(',')
            integer_map = map(int, string_list)
            integer_list = list(integer_map)
            if value is None:
                plt.plot(np.arange(integer_list[0], integer_list[1])/fs, audio[integer_list[0]: integer_list[1], 0],
                         color='grey')
            else:
                plt.plot(np.arange(integer_list[0], integer_list[1])/fs, audio[integer_list[0]: integer_list[1], 0],
                         color='limegreen')
        custom_lines = [Line2D([0], [0], color='grey', lw=4),
                        Line2D([0], [0], color='limegreen', lw=4)]
        plt.legend(custom_lines, ['too quiet', 'ok'])
        plt.title(f'Audio analysis by frame, with quiet threshold (={quiet_threshold}) constraint')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (in seconds)')
        plt.show()

        n_peaks = 0
        for key, value in freqs_by_time_range.items():
            # print(str(key) + ': ' + str(value))
            if value is not None and len(value) > 0: n_peaks += 1
        print(f"{n_peaks} frames had a frequency peak in the frequency range: "
              f"{freq_range if freq_range is not None else [0, fs/2]} Hz.")
    return X, freqs_by_time_range


def stft_main_freqs(data, fs, freqs_by_time_range, N=1024):
    data = data[:, 0]/max(data[:, 0])
    f, t, Zxx = signal.stft(data, fs, nperseg=N)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=max(data)/10)
    num_valid_frames = 0
    for key, value in freqs_by_time_range.items():
        if value is not None:
            num_valid_frames += 1
            string_list = key.split(',')
            integer_map = map(int, string_list)
            integer_list = list(integer_map)
            for v in value:
                audio_time_range = np.arange(integer_list[0], integer_list[1]) / fs
                plt.plot(audio_time_range, [v] * len(audio_time_range), color='white')
    plt.title('Main frequency by time range')
    plt.ylabel('Frequency in Hz')
    plt.xlabel('Time in seconds')
    plt.show()

    print(f'{num_valid_frames} valid frames were analysed.')
    print(f'{len(freqs_by_time_range) - num_valid_frames} frames were discarded because too quiet.')


def find_most_prom_peak_index(P_MUSIC_array, peak_search_range, n=1, prominence=0.5):
    # For each P_MUSIC output in P_MUSIC_array:
    # - Call scipy.find_peaks to find the peak indices
    # - Fill the prominences array with the peak prominences (values determining how peaky are the peaks)
    prominences = []
    for p_music in P_MUSIC_array:
        peaks, _ = sc.find_peaks(p_music, prominence=prominence)
        prominences.append(sc.peak_prominences(p_music, peaks)[0])

    # Fill another array with the same value but this time:
    # - Retrieve only the first prominence (there were two by P_MUSIC output before, because symmetry)
    # - Put None when the array in prominences was empty
    prom = []
    for p in prominences:
        if len(p) > 0:
            prom.append(p[0])
        else:
            prom.append(0)

    # Compute the n maximum prominences in the prom array, and the indices of their corresponding P_MUSIC outputs in
    # P_MUSIC_array
    best_p_music_indices = (-np.array(prom)).argsort()[:n]

    angles_found_array = []
    cmap = get_cmap(len(P_MUSIC_array))
    for p_music_i in best_p_music_indices:
        P_MUSIC_w_most_prom_peak = P_MUSIC_array[p_music_i]
        plt.plot(peak_search_range, P_MUSIC_w_most_prom_peak / max(abs(P_MUSIC_w_most_prom_peak)), c=cmap(p_music_i))
        peaks, _ = sc.find_peaks(P_MUSIC_array[p_music_i], prominence=prominence)
        angles_found = peaks * abs(peak_search_range[2] - peak_search_range[1]) + peak_search_range[0]
        if len(angles_found > 0):
            plt.vlines(angles_found, -1, 0, color='black')
            angles_found_array.append(angles_found[-1])
    plt.title('Spacial spectrum')

    return angles_found_array, np.mean(angles_found_array)


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


# Generate data as described in the setup of the basic implementation of the MUSIC for DOA algo
# (with possibility to add a phase for the correlated signals):
def generate_data(M, N, d, wavelen, angles, freqs, var=0.01, phase = False):
    thetas = np.array(angles) / 180 * np.pi
    w = np.array(freqs)*2*np.pi 
    D = np.size(thetas)
    A = np.exp(-1j * 2 * np.pi * d/wavelen * np.kron(np.arange(M), np.sin(thetas)).reshape((M, D)))
    S = 2 * np.exp(1j * (np.kron(w, np.arange(N)).reshape((D, N))))
    if phase:
        phase_diff = np.random.normal(0, np.pi/4, len(thetas))
        phase_diff = np.exp(1j*phase_diff)
        S = (np.multiply(S.T,phase_diff)).T
    Noise = var * np.random.randn(M, N)
    X = np.dot(A, S) + Noise
    return X
