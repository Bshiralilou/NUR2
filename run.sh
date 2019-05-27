#!/bin/bash

echo "Run handin excercises"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "Plots" ]; then
  mkdir Plots
fi

echo "Run problem one.a ..."
python3 one_a.py > poisson_rands.txt

echo "Run problem one.b ..."
python3 one_b.py 

echo "Run problem one.c ..."
python3 one_c.py 

echo "Run problem two ..."
python3 two_a.py

echo "Run problem three ..."
python3 three_a.py

echo "Run problem four.a ..."
python3 four_a.py

echo "Run problem four.b ..."
python3 four_b.py


echo "Generating the pdf"

pdflatex solution.tex
bibtex solution.aux
pdflatex solution.tex
pdflatex solution.tex
