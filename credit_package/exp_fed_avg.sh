b=-1

# Full participation
python run.py -b $b -s 0 -p False &
python run.py -b $b -s 1 -p False &
python run.py -b $b -s 2 -p False &
python run.py -b $b -s 3 -p False &
python run.py -b $b -s 4 -p False &

# Partial, without replace
python run.py -K 5 -b $b -s 0 -p False &
python run.py -K 5 -b $b -s 1 -p False &
python run.py -K 5 -b $b -s 2 -p False &
python run.py -K 5 -b $b -s 3 -p False &
python run.py -K 5 -b $b -s 4 -p False &

# Partial, with replace
# -r: replace=True
python run.py -K 5 -r True -b $b -s 0 -p False &
python run.py -K 5 -r True -b $b -s 1 -p False &
python run.py -K 5 -r True -b $b -s 2 -p False &
python run.py -K 5 -r True -b $b -s 3 -p False &
python run.py -K 5 -r True -b $b -s 4 -p False &
