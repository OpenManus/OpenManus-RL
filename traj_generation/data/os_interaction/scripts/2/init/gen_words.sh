#!/usr/bin/env bash

declare -a vocab=('aa' 'aaa' 'ab' 'abc' 'able' 'abut' 'ace' 'ache' 'act' 'acm')
declare -a sep=(' ' '  ' '   ' '    ')

out='/usr/words.txt'

echo -n "${vocab[RANDOM % 10]}" > "${out}"
for i in {1..99}; do
  echo -n "${sep[RANDOM % 4]}${vocab[RANDOM % 10]}" >> "${out}"
done
