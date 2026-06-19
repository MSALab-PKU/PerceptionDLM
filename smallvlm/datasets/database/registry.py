
DATASETS = {}
NAMING_DATASETS = []
ANON_DATASETS = []


def register_dataset(naming=False):
    def registering(DB):
        data_type = DB.data_type.lower() if DB.data_type is not None else DB.__name__.lower()
        DATASETS[data_type] = DB
        if naming:
            name = data_type.split('_')
            NAMING_DATASETS.append((name, DB))
        else:
            ANON_DATASETS.append(DB)

        return DB

    return registering

