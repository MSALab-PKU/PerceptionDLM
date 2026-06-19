import warnings
from .registry import DATASETS, NAMING_DATASETS, ANON_DATASETS

def load_database(path, data_type='auto', **kwargs):
    if data_type == 'hf':
        from datasets import load_dataset
        kwargs.update(kwargs.pop('hf_config', {}))
        database = load_dataset(path, **kwargs)
    else:
        DB = parse_database(path, data_type=data_type, **kwargs)
        database = DB(path, **kwargs)

    return database


def parse_database(path: str, data_type='auto', **kwargs):
    if data_type == 'auto':
        DB = auto_database(path, **kwargs)
    elif isinstance(data_type, str):
        data_type = data_type.lower()
        if data_type not in DATASETS:
            raise KeyError(f"Unsupport dataset {data_type}. Try data_type='auto' for unknown dataset.")
        DB = DATASETS[data_type]
    else:
        raise TypeError(f"Unsupport dataset type: {type(data_type)}")

    return DB


def auto_database(path: str, **kwargs):
    DB = type(None)
    path_ = path.lower().replace('-', '').replace('_', '')
    ORDERED_NAMING_DATASETS = sorted(NAMING_DATASETS, key=lambda x: sum(len(w) for w in x[0]), reverse=True)
    for name, DB_ in ORDERED_NAMING_DATASETS:
        if all(word in path_ for word in name) and DB_.pre_check(path, **kwargs):
            DB = DB_
            break
    else:
        for DB_ in ANON_DATASETS:
            if DB_.pre_check(path, **kwargs):
                DB = DB_
    return DB
