// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.ziputils;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * A simple file system abstraction, for testing. This library doesn't itself open files.
 * Clients are responsible for opening files and pass in file channels to the API.
 * This class is currently here to be able to use an common abstraction for testing the
 * library, and tools implemented using this library. This class may be removed in the future.
 */
class FileSystem  {

  protected static FileSystem fileSystem;

  /**
   * Returns the configured file system implementation. If no filesystem is configured, and
   * instance of this class is created and returned. The default filesystem implementation is
   * a simple wrapper of the standard Java file system.
   */
  public static FileSystem fileSystem() {
    if (fileSystem == null) {
      fileSystem = new FileSystem();
    }
    return fileSystem;
  }

  /**
   * Opens a file for reading, and returns a file channel.
   *
   * @param filename name of file to open.
   * @return file channel open for reading.
   * @throws java.io.IOException
   */
  public FileChannel getInputChannel(String filename) throws IOException {
    return Files.newInputStream(Paths.get(filename)).getChannel();
  }

  /**
   * Opens a file for writing, and returns a file channel.
   *
   * @param filename name of file to open.
   * @param append whether to open file in append mode.
   * @return  file channel open for write.
   * @throws java.io.IOException
   */
  public FileChannel getOutputChannel(String filename, boolean append) throws IOException {
    return Files.newOutputStream(Paths.get(filename), append).getChannel();
  }

  /**
   * Opens a file for reading, and returns an input stream.
   *
   * @param filename name of file to open.
   * @return input stream reading from the specified file.
   * @throws java.io.IOException
   */
  public InputStream getInputStream(String filename) throws IOException {
    return Files.newInputStream(Paths.get(filename));
  }
}
