# Install dependencies
pip install -r requirements.txt
# Configure log dir for MBPO
mkdir mbpo_data
echo >> src/defaults.py
echo "ROOT_DIR = '$(pwd)/mbpo_data'" >> src/defaults.py
# Configure force
mkdir force_data
mkdir ~/.force
touch ~/.force/machine.json
echo "{\"default-partition\": 2, \"root-dir\": \"$(pwd)/force_data}\" > ~/.force/machine.json
