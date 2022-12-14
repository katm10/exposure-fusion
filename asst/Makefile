#
# Modify HALIDE_DIR to the appropriate path on your machine.
#
# Special instructions for Mac users
# ==================================
# You need to or must have installed libpng through Macports or Homebrew.
# Assuming that the installation succeeded, you should be able to run
#
# The brew command for installing libpng is
# brew install libpng
#
# libpng-config --I_opts
# libpng-config --L_opts
#
# Please add the output of the above commands to the following variables:
# PNG_INC
# PNG_LIB
#

MKDIR	:= mkdir -p
RM		:= rm -f
CP		:= cp -f
CXX		:= g++ -std=c++17

HALIDE_DIR ?= $(HOME)/Documents/MIT/UROP_FA21/Halide
HALIDE_LIB := $(HALIDE_DIR)/bin/libHalide.so
HALIDE_SRC := $(HALIDE_DIR)/src

BUILD_DIR := bin
SRC_DIR = src
INC_DIR = include
TEST_DIR = test
TUTORIAL_DIR = tutorial

INC  := $(wildcard  $(INC_DIR)/*.h)
SRC  := $(wildcard  $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC))

all: a9 $(OBJECTS)
	mkdir -p Output

CXXFLAGS := -I$(HALIDE_DIR)/include/ -I$(HALIDE_DIR)/tools/ -I. -g -Wall -I$(HALIDE_SRC)/ -I$(INC_DIR)
LDFLAGS  := -L$(HALIDE_DIR)/bin/     -lz -lpthread -ldl -lncurses -lpng -ljpeg

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

MAIN = a9_main.cpp

a9: $(MAIN) $(HALIDE_LIB) $(OBJECTS) $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CFLAGS) $(MAIN) $(OBJECTS) $(HALIDE_LIB) $(LDFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) -rf *.dSYM
	$(RM) -rf $(BUILD_DIR)/*
	rm -rf Output fixed