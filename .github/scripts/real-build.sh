# abort if error
set -e
pip install -r requirements-build.txt
python setup.py bdist_wheel
