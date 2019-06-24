echo "Run handin excercises"

echo "Run problem one.a..."
python3 one_a.py

echo "Run problem one.b..."
python3 one_b.py

echo "Run problem one.c..."
python3 one_c.py

echo "Run problem two..."
python3 two_a.py

echo "Run problem three..."
python3 three_a.py

echo "Run problem four.a..."
python3 four_a.py > integral_4a.txt

echo "Generating the pdf"

pdflatex solutions.tex
bibtex solutions.aux
pdflatex solutions.tex
pdflatex solutions.tex
