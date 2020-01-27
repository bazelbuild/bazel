// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;

import com.google.devtools.build.singlejar.OptionFileExpander.OptionFileProvider;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * A simple virtual file system interface. It's much simpler than the Blaze
 * virtual file system and only to be used inside this package.
 */
public interface SimpleFileSystem extends OptionFileProvider {

  @Override
  InputStream getInputStream(String filename) throws IOException;

  /**
   * Opens a file for output and returns an output stream. If a file of that
   * name already exists, it is overwritten.
   */
  OutputStream getOutputStream(String filename) throws IOException;

  /**
   * Returns the File object for this filename.
   */
  File getFile(String filename) throws IOException;

  /** Delete the file with the given name and return whether deleting it was successful. */
  boolean delete(String filename);
}