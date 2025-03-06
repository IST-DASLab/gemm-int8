# abort if error
set -e
pip3 install -r requirements-build.txt
python3 setup.py bdist_wheel
