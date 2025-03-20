from sys import argv

def analysis_size(size_str):
    size_str = size_str.strip()
    availables = {
        "B": 1,
        "Byte": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024*1024,
        "MB": 1024*1024,
        "G": 1024*1024*1024,
        "GB": 1024*1024*1024,
        "T": 1024*1024*1024*1024,
        "TB": 1024*1024*1024*1024,
        "P": 1024*1024*1024*1024*1024,
        "PB": 1024*1024*1024*1024*1024,        
    }
    for size_unit in availables:
        if size_str.endswith(size_unit):
            return int(size_str[:-len(size_unit)]) * availables[size_unit]
    return int(size_str)

if analysis_size(argv[1]) == analysis_size(argv[2]): 
    exit(0)
exit(1)