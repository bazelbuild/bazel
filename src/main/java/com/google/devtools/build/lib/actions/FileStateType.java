// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.actions;

/** An enum indicating the type of a path on the file system. */
public enum FileStateType {
  REGULAR_FILE("file"),
  /**
   * A special file such as a socket, fifo, or device. See
   * {@link com.google.devtools.build.lib.vfs.FileStatus#isSpecialFile}.
   */
  SPECIAL_FILE("special file"),
  DIRECTORY("directory"),
  SYMLINK("symlink"),
  NONEXISTENT("non-existent path");

  private final String name;

  private FileStateType(String name) {
    this.name = name;
  }

  public String getHumanReadableName() {
    return name;
  }

  /** Returns true if this type does not correspond to a non-existent path. */
  public boolean exists() {
    return this != NONEXISTENT;
  }

  /** Returns true if this value corresponds to a symlink. */
  public boolean isSymlink() {
    return this == SYMLINK;
  }

  /**
   * Returns true if this value corresponds to a regular file. If so, its parent directory is
   * guaranteed to exist.
   */
  public boolean isFile() {
    return this == REGULAR_FILE;
  }

  /**
   * Returns true if this value corresponds to a special file. If so, its parent directory is
   * guaranteed to exist.
   */
  public boolean isSpecialFile() {
    return this == SPECIAL_FILE;
  }

  /**
   * Returns true if the file is a directory. If so, its parent directory is guaranteed to exist.
   */
  public boolean isDirectory() {
    return this == DIRECTORY;
  }
}