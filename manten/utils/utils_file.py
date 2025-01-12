def normalize_filename(filename):
    validchars = "-_"
    out = ""
    for c in filename:
        if str.isalpha(c) or str.isdigit(c) or (c in validchars):
            out += c
        else:
            out += "_"
    return out
