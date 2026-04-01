print('Importing required packages...')
import os, argparse, logging
import numpy as np
import breesy
breesy.log.set_log_level("ERROR")
from scripts import utils
from scripts.data_utils import prepare_X_y_train_test, remove_noisy_epochs

logger = logging.getLogger(__name__)
logger.debug(f"Using breesy version {breesy.__version__}")


def load_labels(label_mode: str, data_files: dict[str, list[str]], filepattern: str, dirname: str,
                labelfile: str | None = None, labelcol: str | None = None, filecol: str | None = None) -> dict[str, int]:

    if label_mode == "pattern":

        assert f'{{{utils.EVENT}}}' in filepattern, f'If "--mode pattern", first "--filepattern" must have {{{utils.EVENT}}} placeholder.'
        labels = utils.extract_labels_from_filenames(
            data_files=data_files, filepattern=filepattern
        )
    elif label_mode == "single":
        assert labelfile, f'"--labelfile" required with "--mode single".'
        assert labelcol, f'"--labelcol" required with "--mode single".'
        assert filecol, f'"--filecol" required with "--mode single".'
        labels = utils.extract_labels_from_table(data_files=data_files, tablename=labelfile, dirname=dirname,
                                                 column_names={"label": labelcol, "file": filecol})
    else:
        parser.error('--mode only supports "single" and "pattern" values for now')

    logger.info(f'Collected {len(data_files)} files with labels.')
    return labels


def main(args: argparse.Namespace, data_savename: str, labels_savename: str) -> None:
    data_files = utils.collect_data_files_by_patterns(datadir=args.datadir, filepatterns=args.filepatterns,
                                                      subjects=args.subjects)
    assert len(data_files) > 0, 'Did not find any files.'
    labels = load_labels(label_mode=args.mode, data_files=data_files, filepattern=args.filepatterns[0],
                         dirname=args.datadir, labelfile=args.labelfile, labelcol=args.labelcol, filecol=args.filecol)

    if args.testid:
        train_filenames, test_filenames = utils.separate_train_test_files(
            data_files=data_files, testid=args.testid, filepattern=args.filepatterns[0]
        )
    else:
        train_filenames = data_files.keys()
        test_filenames = None

    X_train, y_train, X_test, y_test = prepare_X_y_train_test(
        data_files=data_files, labels=labels, test_filenames=test_filenames,
        sample_rates=args.samplerates, epoch_len_s=args.epochlen,
        lowpasses=args.lowpass, highpasses=args.highpass, dirname=args.datadir
    )
    logger.info(f'Shape of a single epoch is {X_train.shape[1:]}')

    X_train, y_train, X_test, y_test = remove_noisy_epochs(X_train=X_train, y_train=y_train,
                                                           X_test=X_test, y_test=y_test, k=args.k)
    
    np.save(data_savename, X_train)
    np.save(labels_savename, y_train)
    print(f'Saved as:\n\t{data_savename}\n\t{labels_savename}\n')
    if X_test is not None:
        test_data_savename = data_savename.rsplit('.', 1)[0] + '_test.npy'
        test_labels_savename = labels_savename.rsplit('.', 1)[0] + '_test.npy'
        np.save(test_data_savename, X_test)
        np.save(test_labels_savename, y_test)
        print(f'Saved test files as:\n\t{test_data_savename}\n\t{test_labels_savename}\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='data preprocessing script for opti-EEG',
        description='A script to preprocess and prepare data for opti-EEG pipeline. Defaults are for the DEED dataset.',
    )
    # paths
    parser.add_argument('datadir', help="Path to directory with data files.")
    parser.add_argument('-o', '--outname', required=True, help='Name to use for saving files.')
    parser.add_argument('-m', '--mode', choices=['pattern', 'single', 'each'], required=True,
                        help='Label reading mode. '
                             '"pattern": labels stored inside file names. '
                             '"single": all labels stored in a single CSV file. '
                             '"each": labels stored in separate files, one per EEG file.')
    parser.add_argument('--labelfile',
                        help='Used with "--mode single". Path to CSV file with labels.') 
    parser.add_argument('-f', '--filepatterns', nargs='+',
                        help='One or more patterns of data files to read from datadir. '
                             'First pattern will be used to extract information like subject ID ({s}). '
                             'If used with "--mode pattern", labels will be read from {e} in the first pattern. '
                             'Provide more than pattern (separated by spaces) to join several files together (e.g. EEG and EMG).')
    parser.add_argument('--labelcol', help='Used with "--mode single". Name of column where labels are stored.')
    parser.add_argument('--filecol', help='Used with "--mode single". Name of column where filenames are stored.')   
    # data & split params
    parser.add_argument('-e', '--epochlen', type=int, help='Length of epochs to cut, in seconds.')
    parser.add_argument('-s', '--samplerates', type=int, nargs='+',
                        help='One or more sample rates, in Hz. Provide as many as there are "--filepatterns".')
    parser.add_argument('-k', type=int, default=4, help='Parameter controlling the ratio of removed epochs with high standard deviation. Higher k = less epochs removed.')
    parser.add_argument('--highpass', type=float, nargs='*', default=[],
                        help='Highpass IIR filter frequencies (Hz). Provide as many numbers as there are "--filepatterns", or not provide at all to disable. Use "0" to disable it for a certain filepattern.')
    parser.add_argument('--lowpass', type=float, nargs='*', default=[],
                        help='Lowpass FIR filter frequencies (Hz). Provide as many as numbers there are "--filepatterns", or not provide at all to disable. Use "0" to disable it for a certain filepattern.')
    parser.add_argument('--subjects', help='Comma-separated subject IDs to load.')
    parser.add_argument('--testid', help='Comma-separated subject IDs to use as test (all should also be present in "--subjects").')

    args = parser.parse_args()

    assert os.path.exists(args.datadir), f"Provided data directory does not exist ({args.datadir})"
    assert os.path.isdir(args.datadir), f"Provided path is not a directory ({args.datadir})"

    if args.filepatterns is not None:
        n_filepatterns = len(args.filepatterns)
        assert (not args.highpass) or (len(args.highpass) == n_filepatterns), "If using highpass, should provide as many numbers as there are filepatterns"
        assert (not args.lowpass) or (len(args.lowpass) == n_filepatterns), "If using lowpass, should provide as many numbers as there are filepatterns"
        assert len(args.samplerates) == n_filepatterns, "Should provide as many sample rate numbers as there are filepatterns"

    data_savename = os.path.join('data', f"{args.outname}_data.npy")
    labels_savename = os.path.join('data', f"{args.outname}_labels.npy")
    os.makedirs('data', exist_ok=True)
    assert not os.path.exists(data_savename), f'Error: file named "{data_savename}" already exists.'
    assert not os.path.exists(labels_savename), f'Error: file named "{labels_savename}" already exists.'

    main(args=args, data_savename=data_savename, labels_savename=labels_savename)
