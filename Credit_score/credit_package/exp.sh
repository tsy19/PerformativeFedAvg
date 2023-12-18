b=4

# Full participation
python run.py -b $b -s 0 &
python run.py -b $b -s 1 &
python run.py -b $b -s 2 &
python run.py -b $b -s 3 &
python run.py -b $b -s 4 &

# Partial, without replace
python run.py -K 5 -b $b -s 0 &
python run.py -K 5 -b $b -s 1 &
python run.py -K 5 -b $b -s 2 &
python run.py -K 5 -b $b -s 3 &
python run.py -K 5 -b $b -s 4 &

# Partial, with replace
# -r: replace=True
python run.py -K 5 -r True -b $b -s 0 &
python run.py -K 5 -r True -b $b -s 1 &
python run.py -K 5 -r True -b $b -s 2 &
python run.py -K 5 -r True -b $b -s 3 &
python run.py -K 5 -r True -b $b -s 4 &
