# abort if error
set -e
# pip3 install -r requirements-build.txt
pip3 install .
python3 setup.py bdist_wheel
