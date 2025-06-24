# Assert Python version exactly 3.8.20
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
REQUIRED_VERSION="3.8.20"
if [ "$PYTHON_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "Python $REQUIRED_VERSION is required. Found Python $PYTHON_VERSION."
    exit 1
fi

python3 -m pip install "setuptools<59"
python3 -m pip install "pip<24.1"
python3 -m pip install "wheel<0.40.0"

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip install "protobuf<3.21.0"

cd abides-core
python3 setup.py install
cd ../abides-markets
python3 setup.py install
cd ../abides-gym
python3 setup.py install
cd ..


export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"