import os
import glob


def find_files(path):
    if os.path.isdir(path):
        path = os.path.join(path, '*')
    files, ext = [], None
    for fp in sorted(glob.glob(path)):
        if os.path.isdir(fp):
            ext_ = '/'
        else:
            ext_ = os.path.splitext(fp)[1]
        if ext is None:
            ext = ext_
        elif ext_ != ext:
            raise ValueError(f"Data file extensions are not same in dir {path}: {ext_} != {ext}. "
                             f"Please keep them the same when loading DataBase from a dir")
        files.append(fp)

    return files


def alphanum_path_key(s: str):
    """Use to sort string by numbers in it, useful when numbers in string are not matched in digit (1 and 001)."""
    s = os.path.splitext(s)[0]
    digit_key_list = []
    str_section = ''
    is_digit = False
    for c in s:
        if c.isdigit() == is_digit:
            str_section += c
        else:
            if is_digit:
                digit_key_list.append(int(str_section))
            else:
                digit_key_list.append(str_section)
            str_section, is_digit = c, c.isdigit()
    if is_digit:
        digit_key_list.append(int(str_section))
    else:
        digit_key_list.append(str_section)
    return digit_key_list
