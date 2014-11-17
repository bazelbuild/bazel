// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import java.io.IOException;

/**
 * Allow tests to easily manage scratch files in a FileSystem.
 */
public final class Scratch {

  private final FileSystem fileSystem;

  /**
   * Create a new ScratchFileSystem using the {@link InMemoryFileSystem}
   */
  public Scratch() {
    this(new InMemoryFileSystem(BlazeClock.instance()));
  }

  /**
   * Create a new ScratchFileSystem using the supplied FileSystem.
   */
  public Scratch(FileSystem fileSystem) {
    this.fileSystem = fileSystem;
  }

  /**
   * Returns the FileSystem in use.
   */
  public FileSystem getFileSystem() {
    return fileSystem;
  }

  /**
   * Create a directory in the scratch filesystem, with the given path name.
   */
  public Path dir(String pathName) throws IOException {
    Path dir = getFileSystem().getPath(pathName);
    if (!dir.exists()) {
      FileSystemUtils.createDirectoryAndParents(dir);
    }
    if (!dir.isDirectory()) {
      throw new IOException("Exists, but is not a directory: " + pathName);
    }
    return dir;
  }

  /**
   * Create a scratch file in the scratch filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  public Path file(String pathName, String... lines)
      throws IOException {
    Path newFile = file(getFileSystem(), pathName, lines);
    newFile.setLastModifiedTime(-1L);
    return newFile;
  }

  /**
   * Like {@code scratchFile}, but the file is first deleted if it already
   * exists.
   */
  public Path overwriteFile(String pathName, String... lines) throws IOException {
    Path oldFile = getFileSystem().getPath(pathName);
    long newMTime = oldFile.exists() ? oldFile.getLastModifiedTime() + 1 : -1;
    oldFile.delete();
    Path newFile = file(getFileSystem(), pathName, lines);
    newFile.setLastModifiedTime(newMTime);
    return newFile;
  }

  /**
   * Deletes the specified scratch file, using the same specification as {@link Path#delete}.
   */
  public boolean deleteFile(String pathName) throws IOException {
    Path file = getFileSystem().getPath(pathName);
    return file.delete();
  }

  /**
   * Create a scratch file in the given filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  public Path file(FileSystem fs, String pathName, String... lines)
      throws IOException {
    Path file = newScratchFile(fs, pathName);
    FileSystemUtils.writeContentAsLatin1(file, linesAsString(lines));
    return file;
  }

  /**
   * Create a scratch file in the given filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  public Path file(FileSystem fs, String pathName, byte[] content)
      throws IOException {
    Path file = newScratchFile(fs, pathName);
    FileSystemUtils.writeContent(file, content);
    return file;
  }

  /** Creates a new scratch file, ensuring parents exist. */
  private Path newScratchFile(FileSystem fs, String pathName) throws IOException {
    Path file = fs.getPath(pathName);
    Path parentDir = file.getParentDirectory();
    if (!parentDir.exists()) {
      FileSystemUtils.createDirectoryAndParents(parentDir);
    }
    if (file.exists()) {
      throw new IOException("Could not create scratch file (file exists) "
          + pathName);
    }
    return file;
  }

  /**
   * Converts the lines into a String with linebreaks. Useful for creating
   * in-memory input for a file, for example.
   */
  private static String linesAsString(String... lines) {
    StringBuilder builder = new StringBuilder();
    for (String line : lines) {
      builder.append(line);
      builder.append('\n');
    }
    return builder.toString();
  }
}
