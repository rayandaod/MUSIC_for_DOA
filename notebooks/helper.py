import numpy as np
import scipy.signal as sc
import params


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def music_real(X, main_freq, d, D, M, locations):
    # Make sure we have more microphones than input signals
    assert (M > D)

    # Peak search range (start degree, end degree, step)
    peak_search_range = np.arange(-90, 90, 1)
    peak_search_range_size = len(peak_search_range)

    # Compute the wavelength for X
    wavelen = params.SPEED_OF_SOUND / main_freq

    # Put the first mic at location (0,0)
    locations = locations - locations[0, :]

    # Centralize X
    X_mean = np.mean(X, axis=1)
    X_centered = X - np.tile(X_mean, (np.shape(X)[1], 1)).T

    # Covariance matrix
    R = np.dot(X_centered, X_centered.conj().T)

    # Mystery and boule of gomme
    # if M == 8:
    #    J = np.flip(np.eye(M), axis=1)
    #    R = R + np.dot(J, np.dot(R.conj(), J))

    # Eigen value decomposition
    eig_val, eig_vect = np.linalg.eig(R)

    # Find the smallest M-D eigen values (corresponding to the noise subspace) and their eigen vectors
    ids = np.abs(eig_val).argsort()[:(M - D)]
    En = eig_vect[:, ids]

    # Noise subspace estimation: Ren = EnEn'
    Ren = np.dot(En, En.conj().T)

    # if M == 8:
    atheta = np.exp(-1j * 2 * np.pi / wavelen *
                    np.kron(d, np.sin(peak_search_range / 180 * np.pi)).reshape(M, peak_search_range_size))
    # else:
    #    theta = np.arange(0, 360, 1)
    #    a = np.array([np.cos(theta/180*np.pi), np.sin(theta/180*np.pi)])
    #    atheta = np.exp(-1j*2*np.pi/wavelen*np.dot(locations, a))

    # Compute the resulting vector P
    P_MUSIC = np.zeros(peak_search_range_size)
    for j in range(peak_search_range_size):
        P_MUSIC[j] = 1 / abs(np.dot(np.dot(atheta[:, j].conj().T, Ren), atheta[:, j]))

    P_MUSIC = np.log(P_MUSIC / np.max(P_MUSIC))
    return P_MUSIC, peak_search_range


def find_angles(P_MUSIC, peak_search_range, angles_to_be_found, prominence=10, width=1):
    peaks, _ = sc.find_peaks(P_MUSIC, prominence=prominence, width=width)
    peaks = peaks * abs(peak_search_range[2] - peak_search_range[1]) + peak_search_range[0]
    print("Actual angle(s) is/are:", angles_to_be_found)
    print("Angle(s) found is/are:", peaks)
    if len(angles_to_be_found) == len(peaks):
        print(np.sum(np.sort(np.array(peaks)) == np.sort(np.array(angles_to_be_found))), "/", len(peaks),
              "correct guesses")
    else:
        print(f"{bcolors.WARNING}Number of sources and number of peaks do not match{bcolors.ENDC}")
    return peaks


# def find_highest_RMS(signal, frame_size=1000, hop_size=500):
#     assert(len(signal) >= frame_size)
#     rms_array = []
#     for frame in :
#         rms_array.append(np.sqrt(np.mean(frame ** 2)))
#     rms_max,