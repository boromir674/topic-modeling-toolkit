#!/bin/bash

current_dir=$(echo $PWD)
cd $1
git clone https://github.com/bigartm/bigartm.git
cd $current_dir
