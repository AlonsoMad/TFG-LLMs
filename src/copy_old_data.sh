#!/bin/bash

if [ ! -d "/Data/3_joined_data" ]; then
    echo "Copying 3_joined_data..."
    cp -r /Data_old/3_joined_data /Data/
fi

if [ ! -d "/Data/mallet_folder" ]; then
    echo "Copying mallet_folder..."
    cp -r /Data_old/mallet_folder /Data/
fi

exec "$@"
