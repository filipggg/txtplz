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


Examples
--------

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
