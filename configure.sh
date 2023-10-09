# !bin/bash

searchspaces_dropbox_link="https://www.dropbox.com/s/o4mkf1ueut51oy2/searchspaces.zip?dl=0"

if [ -d "searchspaces" ]; then
  echo "The 'searchspaces' folder already exists."
else
  echo "The 'searchspaces' folder does not exist. Downloading..."
  wget "$searchspaces_dropbox_link" -q --no-check-certificate -O searchspaces.zip --show-progress
  unzip searchspaces.zip
  rm -rf __MACOSX
  rm searchspaces.zip
  echo "Download complete."
fi

# Check if the "thesisenv" environment exists
if conda env list | grep -q "thesisenv"; then
    echo "The 'thesisenv' environment already exists."
else
    # Create the Conda environment
    conda create -n thesisenv python=3.10.8 -y
    
    # Activate the environment
    conda activate thesisenv
    
    # Install packages from requirements.txt
    pip install -r requirements.txt
    
    echo "The 'thesisenv' environment has been created."
fi