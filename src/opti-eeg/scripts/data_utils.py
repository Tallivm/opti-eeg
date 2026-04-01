import os, logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from scipy import signal as si
import breesy

logger = logging.getLogger(__name__)


# Loading raw data

def load_cat_split_recording(filename: str, other_names: list[str], sample_rates: list[int],
                             epoch_len_s: int | float, lowpasses: list[float], highpasses: list[float],
                             dirname: str) -> list[np.ndarray]:
    rec = breesy.load.load_recording(filename, sample_rate=sample_rates[0])
    if lowpasses:
        frequency = lowpasses[0]
        transition_width = frequency * 0.15
        rec = breesy.filtering.lowpass_filter_fir(recording=rec, frequency=frequency, transition_width=transition_width)
    if highpasses:
        rec = breesy.filtering.highpass_filter_iir(recording=rec, frequency=frequency, filter_order=5)

    lowest_sr = min(sample_rates)
    if rec.sample_rate > lowest_sr:
        rec = breesy.processing.downsample(rec, new_sample_rate=lowest_sr)

    if other_names:
        to_concat = [rec.data]
        for name, sr, low, high in zip(other_names, sample_rates[1:], highpasses[1:], lowpasses[1:]):
            add = breesy.load.load_recording(os.path.join(dirname, name), sample_rate=sr)
            add_filtered = breesy.filtering.bandpass_filter(recording=add, low=low, high=high)
            if add_filtered.sample_rate > lowest_sr:
                add_filtered = breesy.processing.downsample(add_filtered, new_sample_rate=lowest_sr)
            to_concat.append(add_filtered.data)
        least_samples = min(x.shape[-1] for x in to_concat)
        data = np.concat([x[:, :least_samples] for x in to_concat], axis=0)
        rec = breesy.recording.Recording(
            data=data, channel_names=range(data.shape[0]), sample_rate=lowest_sr
        )

    epochs = breesy.recording.split_by_window_duration(
        recording=rec, window_duration=epoch_len_s, overlap_duration=0
    )
    return [e.data for e in epochs]


def load_and_split_recording(filename: str, sample_rate: int, epoch_len: int | float,
                             bandpass: tuple[int | float | None, int | float | None]) -> list[np.ndarray]:
    rec = breesy.load.load_recording(filename, sample_rate=sample_rate)
    filtered = breesy.filtering.bandpass_filter(recording=rec, low=bandpass[0], high=bandpass[1])
    epochs = breesy.recording.split_by_window_duration(
        recording=filtered, window_duration=epoch_len/sample_rate, overlap_duration=0
    )
    return [e.data for e in epochs]


def prepare_X_y_train_test(data_files: dict[str, list[str]], labels: dict[str, str],
                           test_filenames: list[str], sample_rates: list[int],
                           epoch_len_s: int | float, lowpasses: list[int | float], highpasses: list[int | float],
                           dirname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    X_train, y_train, X_test, y_test = [], [], [], []
    for filename in tqdm(data_files.keys(), total=len(data_files), desc="Loading epochs"):
        epochs = load_cat_split_recording(filename=filename, other_names=data_files[filename],
                                          sample_rates=sample_rates, epoch_len_s=epoch_len_s,
                                          lowpasses=lowpasses, highpasses=highpasses, dirname=dirname)
        if test_filenames is not None and (filename in test_filenames):
            X_test.extend(epochs)
            y_test.extend([labels[filename] for _ in epochs])            
        else:
            X_train.extend(epochs)
            y_train.extend([labels[filename] for _ in epochs])

    X_train, y_train = np.array(X_train), np.array(y_train)
    if test_filenames:
        X_test, y_test = np.array(X_test), np.array(y_test)
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_test, y_test

# Loading processed data

def load_and_process_npy(data_path: str, ch_names: list[str], omit_ch_names: list[str]) -> np.ndarray:
    data = np.load(data_path)
    assert data.ndim == 3, f'Data has incorrect shape, ndim should be 3 but found {data.ndim}'
    assert not np.isnan(data).any(), "NaN (missing) values detected in data!"
    data = drop_channels(data=data, ch_names=ch_names, to_omit=omit_ch_names, ndim=3)

    return data


def load_data(data_path: str, labels_path: str, test_data_path: str | None, test_labels_path: str | None,
              ch_names: list[str], omit_ch_names: list[str],
              class_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    assert data_path.endswith(".npy"), "Only .npy files are currently supported"

    data = load_and_process_npy(data_path=data_path, ch_names=ch_names, omit_ch_names=omit_ch_names)

    if data.dtype == 'float64':
        logger.warning('Data has high precision (float64). It will be lowered to float32.')
    labels = load_labels_from_npy(labels_path=labels_path)
    data, labels, n_cls = remap_labels(data=data, labels=labels, label_names=class_names)

    if test_data_path and test_labels_path:
        logging.info("Loading test files...")
        test_data = load_and_process_npy(data_path=test_data_path, ch_names=ch_names, omit_ch_names=omit_ch_names)
        test_labels = load_labels_from_npy(labels_path=test_labels_path)
        test_data, test_labels, _ = remap_labels(data=test_data, labels=test_labels, label_names=class_names)
    else:
        test_data, test_labels = None, None

    return data, labels, test_data, test_labels, n_cls


def load_labels_from_npy(labels_path: str) -> np.ndarray:
    arr = np.load(labels_path)
    assert arr.ndim == 1, f"Only 1D labels supported, got ndim={arr.ndim}."
    return arr.astype(str)


def get_train_test(data: np.ndarray, labels: np.ndarray, seed: int,
                   ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = train_test_split(data, labels, random_state=seed, stratify=labels, test_size=ratio)
    X_train, X_test, y_train, y_test = split
    return X_train, X_test, y_train, y_test


def load_and_validate_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    # load data
    data_path = config["path"]["data_path"]
    labels_path = config["path"]["labels_path"]
    logger.debug(f'Data & labels will be loaded from:\n\t{data_path}\n\t{labels_path}')
    if config["path"]["test_data_path"]:
        test_data_path = config["path"]["test_data_path"]
        test_labels_path = config["path"]["test_labels_path"]
        logger.debug(f'Test data & labels will be loaded from:\n\t{test_data_path}\n\t{test_labels_path}')
    else:
        logger.debug(f'Test data & labels will be split from all data.')
        test_data_path, test_labels_path = None, None

    data, labels, test_data, test_labels, n_cls = load_data(
        data_path=data_path,
        labels_path=labels_path,
        test_data_path=test_data_path,
        test_labels_path=test_labels_path,
        ch_names=config["data"]["channel_names"],
        omit_ch_names=config["data"]["omit_channels"],
        class_names=config["data"]["class_names"],
    )
    config["data"]["n_classes"] = n_cls
    logger.info(f'Loaded & processed data shape: {data.shape}')
    logger.info(f'Loaded labels shape: {labels.shape}')
    if test_data is not None and test_labels is not None:
        logger.info(f'Test data - Loaded & processed data shape: {test_data.shape}')
        logger.info(f'Test data - Loaded labels shape: {test_labels.shape}')

    if test_data is None:
        train_data, test_data, train_labels, test_labels = get_train_test(
            data=data, labels=labels, seed=config["train"]["data_seed"], ratio=0.1
        )
    else:
        train_data, train_labels = data, labels

    return train_data, train_labels, test_data, test_labels


# Processing raw data

def remap_labels(data: np.ndarray, labels: np.ndarray, label_names: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    drop_labels = []
    remap_labels = {}
    for i, label in enumerate(label_names):
        if label == "":
            drop_labels.append(i)
        elif label in label_names[:i]:
            remap_labels[i] = label_names[:i].index(label)
    logger.debug(f'Will drop labels: {drop_labels}')
    logger.debug(f'Will remap labels: {remap_labels}')

    for to_drop in drop_labels:
        keep_pos = labels != to_drop
        data = data[keep_pos]
        labels = labels[keep_pos]
    for remap_from, remap_to in remap_labels.items():
        labels[labels == remap_from] = remap_to

    label_ints, new_labels, label_counts = np.unique(labels, return_inverse=True, return_counts=True)
    logger.info('Prepared class labels and counts:')
    for i, (v, c) in enumerate(zip(label_ints, label_counts)):
        try:
            class_name = label_names.index(v)
        except ValueError:
            raise ValueError(f'File has label {v} but it is not in the list of labels in config ({label_names}).')
        logger.info(f'\t{i} ({class_name}, {v} in data file): {c}')

    return data, new_labels, len(label_ints)


# Processing preprocessed lata

def drop_channels(data: np.ndarray, ch_names: list[str], to_omit: list[str], ndim: int) -> np.ndarray:
    assert ndim in [2, 3], f"Number of data dimensions must be either 2 or 3, got {ndim}"
    assert data.shape[-2] == len(ch_names), f'Channel name list has {len(ch_names)} channels, should be {data.shape[-2]} channels'
    assert len(to_omit) < len(ch_names), 'Requested to drop all channels - there is a mistake somewhere.'
    channels_to_keep = [i for i, x in enumerate(ch_names) if x not in to_omit]
    if ndim == 3:
        return data[:, channels_to_keep, :]
    if ndim == 2:
        return data[channels_to_keep, :]


def bandpass_filter(data: np.ndarray, bandpass: tuple[int | float | None, int | float | None], sample_rate: int) -> np.ndarray:
    if (not bandpass[0]) and (not bandpass[1]):
        return data
    if bandpass[0] is not None and bandpass[1] is not None:
        assert bandpass[0] < bandpass[1], f'Highpass value should be lower than lowpass value, got {bandpass[0]} and {bandpass[1]}'
    else:
        raise ValueError('Only full bandpass (highpass+lowpass) filtering currently supported, please provide both or none.')

    numtaps = int(sample_rate / 10.0 * 3.3)  # TODO: smoothness_factor=10.0 different due to different length of epochs!
    numtaps = numtaps if numtaps % 2 == 1 else numtaps + 1
    fir_coeff = si.firwin(numtaps=numtaps, cutoff=[bandpass[0], bandpass[1]],
                          pass_zero=False, fs=sample_rate, window="hamming")
    filtered = si.filtfilt(b=fir_coeff, a=[1.0], x=data, axis=-1)
    return filtered


def remove_noisy_epochs(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray | None, y_test: np.ndarray | None,
                        k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    stds = X_train.std(axis=(1, 2))
    thresh = get_artifact_threshold(epoch_sds=stds, k=k)
    X_train_good = X_train[stds<thresh]
    y_train_good = y_train[stds<thresh]
    print(f'Kept {X_train_good.shape[0]/X_train.shape[0]*100:.2f}% epochs with SD < {thresh:.5f}')
    if (X_test is not None) and (y_test is not None):
        test_stds = X_test.std(axis=(1, 2))  # test should not be used for threshold calculation
        X_test_good = X_test[test_stds<thresh]
        y_test_good = y_test[test_stds<thresh]
        print(f'Kept {X_test_good.shape[0]/X_test.shape[0]*100:.2f}% test epochs')
    else:
        X_test_good, y_test_good = None, None
    return X_train_good, y_train_good, X_test_good, y_test_good


# Other

def get_class_weights(train_labels: np.ndarray) -> list[float]:
    class_counts = np.bincount(train_labels).astype(float)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    return [float(x) for x in class_weights]


def standardize_by_train_3D(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray = None) -> list[np.ndarray]:
    assert X_train.ndim == 3, f"Train data must have ndim=3, got {X_train.ndim}"
    m = X_train.mean(axis=(0, 2), keepdims=True)
    s = X_train.std(axis=(0, 2), keepdims=True)
    return [(X - m) / s for X in [X_train, X_val, X_test] if X is not None]


def get_artifact_threshold(epoch_sds: np.ndarray, k: int) -> float:
    """k: 3-5 typical; higher = more lenient. Claude Opus 4.5"""
    median_sd = np.median(epoch_sds)
    mad = np.median(np.abs(epoch_sds - median_sd))
    threshold = median_sd + k * 1.4826 * mad
    return threshold
