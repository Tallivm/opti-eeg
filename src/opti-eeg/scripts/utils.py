import os, re, random, gc, json, logging, csv
from datetime import datetime
from pathlib import Path
import numpy as np
from torch import save as torch_save
from torch import manual_seed as torch_manual_seed
from torch import cuda, get_rng_state
from torch.nn import Module
from torch.optim import Optimizer
from torch.backends import cudnn
from parse import parse

from breesy.load import SUPPORTED_FILE_EXTENSIONS


logger = logging.getLogger(__name__)

SUB_ID = "s"
EVENT = "e"


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
   
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def setup_logger(logger, loglevel: int) -> None:
    logger_level = {'debug': logging.DEBUG,
                    'info': logging.INFO,
                    'warn': logging.WARNING}[loglevel]
    logger.setLevel(logger_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger_level)
    logfile_name = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.log")
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", logfile_name))
    file_handler.setLevel(logger_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def read_config(config_name: str) -> dict:
    assert config_name.lower().endswith(".json"), "Config should be a json file."
    with open(config_name, 'r') as f:
        config = json.load(f)
    return config


def assert_required_params(param_list: list[str], params: dict, callable_name: str) -> None:
    for p in param_list:
        assert p in params.keys(), f'Parameter {p} is required for callable {callable_name}.'


def save_crash_params_dict(param_config: dict, run_name: str) -> None:
    json_filename = os.path.join("logs", f"crash_params_{run_name}.json")
    with open(json_filename, 'x') as f:
        json.dump(param_config, f)


def to_numpy(x):
    if hasattr(x, 'cpu'):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_checkpoint(model: Module, checkpoint_path: str, optimizer: Optimizer, config: dict) -> None:
    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch_save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'hyper_parameters': dict(config),
        'random_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': get_rng_state()
        }
    }, checkpoint_path)


def clean_gpu_memory():
    cuda.empty_cache()
    gc.collect()


# String operations


def remove_dir_from_path(f: str, dir: str | None) -> str:
    if dir is None:
        return Path(f).name
    return f.split(dir)[1].strip('/')


def str_dict_items(title: str, d: dict[str, int | float]) -> str:
    string = title + '\n'
    for k, v in d.items():
        string += f"\t{k}: {v:.4f}\n"
    return string


def validate_file_extension(filepattern: str) -> str:
    assert filepattern, '"--filepattern" cannot be empty.'
    assert '.' in filepattern, f'"--filepattern" should contain file extension.'
    file_extension = '.' + filepattern.rsplit('.', 1)[-1].lower()
    assert file_extension in SUPPORTED_FILE_EXTENSIONS, f'"{file_extension}" files currently not supported.'
    return file_extension


def collect_data_files_by_patterns(datadir: str, filepatterns: str, subjects: str | None) -> dict[str, list[str]]:
    first_pattern = filepatterns[0]
    validate_file_extension(filepattern=first_pattern)

    if subjects:
        assert f'{{{SUB_ID}}}' in first_pattern, f'If not all subjects are taken, "--filepatterns" must have {{{SUB_ID}}} placeholders.'
        subjects = [x for x in subjects.split(',') if x]

    data_files = {str(f): [] for f in Path(datadir).rglob(re.sub(r'\{[^}]+\}', '*', first_pattern))}
    assert len(data_files) > 0, f'Did not find any files with pattern {first_pattern}.'
    logger.debug(f'Found {len(data_files)} files with pattern "{first_pattern}".')

    wrong_subject, missing_files = [], []
    for f in data_files.keys():
        named_templates = parse(first_pattern, remove_dir_from_path(f=f, dir=datadir)).named
        if subjects and (named_templates[SUB_ID] not in subjects):
            wrong_subject.append(f)
            continue

        for pattern in filepatterns[1:]:
            validate_file_extension(filepattern=pattern)
            filled_pattern = pattern.format(**named_templates)
            if os.path.exists(os.path.join(datadir, filled_pattern)):
                data_files[f].append(filled_pattern)
            else:
                missing_files.append(f)
    
    if len(missing_files) > 0:
        logger.warning(f'Did not find all additional files for {len(missing_files)} recordings!')
    data_files = {k: v for k, v in data_files.items()
                  if (k not in missing_files) and (k not in wrong_subject)}
    return data_files


def separate_train_test_files(data_files: dict[str, list[str]], testid: str, filepattern: str) -> tuple[list[str], list[str]]:
    test_subjects = [x for x in testid.split(',') if x]
    train_filenames, test_filenames = [], []
    for filename in data_files.keys():
        if parse(filepattern, filename)[SUB_ID] in test_subjects:
            test_filenames.append(filename)
        else:
            train_filenames.append(filename)
    return train_filenames, test_filenames


def extract_labels_from_filenames(data_files: dict[str, list[str]], filepattern: str) -> dict[str, str]:
    labels = {}
    for filename in data_files.keys():
        labels[filename] = parse(filepattern, remove_dir_from_path(f=filename, dir=None))[EVENT]
    return labels


def extract_labels_from_table(data_files: dict[str, list[str]], tablename: str,
                              column_names: dict, dirname: str) -> dict[str, str]:
    assert tablename.lower().endswith(".csv"), "Only CSV files with labels currently supported."
    labels, not_found = {}, 0

    with open(tablename, newline="") as f:
        reader = csv.DictReader(f)
        filecol = column_names["file"]
        assert filecol in reader.fieldnames, f'Column "{filecol}" not present in file {tablename}.'
        labelcol = column_names["label"]
        assert labelcol in reader.fieldnames, f'Column "{labelcol}" not present in file {tablename}.'
        rows = list(reader)

    filename_to_label = {}
    for row in rows:
        filename = row[filecol]
        label = row[labelcol]
        existing_labels = filename_to_label.get(filename, [])
        filename_to_label[filename] = existing_labels + [label]

    for filename in data_files.keys():
        filename = remove_dir_from_path(f=filename, dir=dirname)
        if filename not in filename_to_label:
            labels[filename] = None
            not_found += 1
            continue
        
        label = filename_to_label[filename]
        assert len(label) == 1, f"Found more than one row for the file {filename} in {tablename}!"  # TODO: instead check if labels are all same, otherwise raise error
        labels[filename] = str(label[0])
        
    if not_found > 0:
        logger.warning(f'{not_found} files are not present in file {tablename}, their labels will be None.')

    return labels
