#!/bin/bash

git submodule init
git submodule update --recursive

cp ./whisper.cpp/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml.c ./cpp/ggml.c
cp ./whisper.cpp/whisper.h ./cpp/whisper.h
cp ./whisper.cpp/whisper.cpp ./cpp/whisper.cpp
cp -R ./whisper.cpp/coreml/ ./cpp/coreml/

yarn example
