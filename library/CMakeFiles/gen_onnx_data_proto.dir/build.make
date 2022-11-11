# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/onceas/Arantir/onnx

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/onceas/Arantir/onnx/.setuptools-cmake-build

# Utility rule file for gen_onnx_data_proto.

# Include any custom commands dependencies for this target.
include CMakeFiles/gen_onnx_data_proto.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gen_onnx_data_proto.dir/progress.make

CMakeFiles/gen_onnx_data_proto: onnx/onnx-data.pb.cc
CMakeFiles/gen_onnx_data_proto: onnx/onnx-data.pb.h

onnx/onnx-data.pb.cc: onnx/onnx-data.proto
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/onceas/Arantir/onnx/.setuptools-cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running C++ protocol buffer compiler on /home/onceas/Arantir/onnx/.setuptools-cmake-build/onnx/onnx-data.proto"
	/usr/bin/protoc /home/onceas/Arantir/onnx/.setuptools-cmake-build/onnx/onnx-data.proto -I /home/onceas/Arantir/onnx/.setuptools-cmake-build --cpp_out dllexport_decl=ONNX_API:/home/onceas/Arantir/onnx/.setuptools-cmake-build --python_out /home/onceas/Arantir/onnx/.setuptools-cmake-build --plugin protoc-gen-mypy=/home/onceas/Arantir/onnx/.setuptools-cmake-build/tools/protoc-gen-mypy.sh --mypy_out dllexport_decl=ONNX_API:/home/onceas/Arantir/onnx/.setuptools-cmake-build

onnx/onnx-data.pb.h: onnx/onnx-data.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate onnx/onnx-data.pb.h

onnx/onnx-data.proto: ../onnx/onnx-data.in.proto
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/onceas/Arantir/onnx/.setuptools-cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Running gen_proto.py on onnx/onnx-data.in.proto"
	/usr/bin/python3 /home/onceas/Arantir/onnx/onnx/gen_proto.py -p onnx -o /home/onceas/Arantir/onnx/.setuptools-cmake-build/onnx onnx-data -m

gen_onnx_data_proto: CMakeFiles/gen_onnx_data_proto
gen_onnx_data_proto: onnx/onnx-data.pb.cc
gen_onnx_data_proto: onnx/onnx-data.pb.h
gen_onnx_data_proto: onnx/onnx-data.proto
gen_onnx_data_proto: CMakeFiles/gen_onnx_data_proto.dir/build.make
.PHONY : gen_onnx_data_proto

# Rule to build all files generated by this target.
CMakeFiles/gen_onnx_data_proto.dir/build: gen_onnx_data_proto
.PHONY : CMakeFiles/gen_onnx_data_proto.dir/build

CMakeFiles/gen_onnx_data_proto.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gen_onnx_data_proto.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gen_onnx_data_proto.dir/clean

CMakeFiles/gen_onnx_data_proto.dir/depend:
	cd /home/onceas/Arantir/onnx/.setuptools-cmake-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/onceas/Arantir/onnx /home/onceas/Arantir/onnx /home/onceas/Arantir/onnx/.setuptools-cmake-build /home/onceas/Arantir/onnx/.setuptools-cmake-build /home/onceas/Arantir/onnx/.setuptools-cmake-build/CMakeFiles/gen_onnx_data_proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gen_onnx_data_proto.dir/depend

