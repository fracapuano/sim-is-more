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
if conda env list | grep -q "oscarenv"; then
    echo "The 'oscarenv' environment already exists."
else
    # Create the Conda environment
    conda create -n oscarenv python=3.11 -y

    # Activate the environment
    conda activate oscarenv

    # Install packages from poetry project
    pip install poetry
    poetry install

    echo "The 'oscarenv' environment has been created and the relevant dependancies have been installed."
fi

