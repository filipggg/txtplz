txtplz
======

Command-line tool for generating texts using neural language models.


Dependencies
------------

* fairseq
* Transformers
* python-regex

### On ArchLinux

* fairseq (not fairseq-git)
* python-sentencepiece
* python-transformers
* python-huggingface-hub


Examples
--------

Just generate some text

```
$ echo 'A programmer, a horse and a GPU enter a bar' | ./txtplz.py gpt2.xl

A programmer and a horse enter a bar.

A programmer says: "I'm interested in computer science. I have a few friends who do it well." "Do you have money to buy a horse?" "I could buy him, but the horse looks too old for riding."

"Hello, I would like your help to put my ideas into a game. Please reply with a program."
 the programmer just looks at me for few seconds to guess if I am really interested
```

Generate some names

```
$ printf '%s\n' {Japanese,Norwegian,Polish,Spanish}$'\t'{death\ metal,hip\ hop,smooth\ jazz} | ./txtplz.py gpt2.xl --input-pattern 'The newly founded {1} {2} band called "' --search '^.*?"[^"]+"' --temperature 1.25 --batch-size 4
The newly founded Japanese death metal band called "Hellhammer."
The newly founded Japanese hip hop band called "Kawade"
The newly founded Japanese smooth jazz band called "Makoto Enshou"
The newly founded Norwegian death metal band called "Lemarchant"
The newly founded Norwegian hip hop band called "We the Robots"
The newly founded Norwegian smooth jazz band called "Norwegian Sky Choir"
The newly founded Polish death metal band called "Death With Black Wings"
The newly founded Polish hip hop band called "Gnocchi and a Kień,"
The newly founded Polish smooth jazz band called "Wunderwudl"
The newly founded Spanish death metal band called "Death From Above"
The newly founded Spanish hip hop band called "Hip Hop Is the Enemy of God and Hip Hop's Opposite"
The newly founded Spanish smooth jazz band called "Los Callejos"
```

Question answering

```
$ cat countries.txt
Poland
Argentina
Senegal
Laos
Wielkopolska
Soviet Union
$ ./txtplz.py gpt2.xl --delimiter '[\.\n,\(]' --input-pattern 'Q: What is the capital city of {}? A:'

Q: What is the capital city of Poland? A: Warsaw
Q: What is the capital city of Argentina? A: Buenos Aires
Q: What is the capital city of Senegal? A: Dakar
Q: What is the capital city of Laos? A: Asiatic
Q: What is the capital city of Wielkopolska? A: Wielin Wielka
Q: What is the capital city of Soviet Union? A: Moscow
```

Lemmatize Polish names

```
$ cat names.txt
Filipa Gralińskiego
Franciszkiem Skąpskim
Marka Smeatona
Adama Małysza
Stanisławem Lemem

$ cat names.txt | ./txtplz.py polish.gpt2.large --prompt 'Podaj formę w mianowniku: Józefa Koronackiego - Józef Koronacki, Anny Budki - Anna Budka, Maciejem Lasoniem - Maciej Lasoń, ' --input-pattern '{} -' --delimiter '[,\.]' --batch-size 4 --remove-input --no-sampling
 Filip Graliński
 Franciszek Skąpanski
 Marek Smetona
 Adam Małysz
 Stanisław Lem
