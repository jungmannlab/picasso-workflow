#!/bin/zsh

current_dir=$(dirname "$0")

source ~/.bash_profile

cd $current_dir

echo Starting picasso-workflow GUI

conda activate picasso-workflow
 
python gui.py

echo Shutting down picasso-workflow GUI
 
conda deactivate