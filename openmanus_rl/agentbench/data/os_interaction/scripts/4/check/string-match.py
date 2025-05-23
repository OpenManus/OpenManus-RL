from sys import argv

def norm_newline(s):
  return s.replace("\r\n", "\n").replace("\r", "\n")

if norm_newline(argv[1]).strip() == norm_newline(argv[2]).strip():
  exit(0)
exit(1)