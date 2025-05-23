from sys import argv

def norm_newline(s):
  return s.replace("\r\n", "\n").replace("\r", "\n")

v1 = norm_newline(argv[1]).strip()
v2 = norm_newline(argv[2]).strip()

if v1 in v2:
  exit(0)
else:
  exit(1)