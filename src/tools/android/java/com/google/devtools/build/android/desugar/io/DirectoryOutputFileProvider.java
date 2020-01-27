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
package com.google.devtools.build.android.desugar.io;

import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/** Output provider is a directory. */
class DirectoryOutputFileProvider implements OutputFileProvider {

  private final Path root;

  public DirectoryOutputFileProvider(Path root) {
    this.root = root;
  }

  @Override
  public void copyFrom(String filename, InputFileProvider inputFileProvider, String outputFilename)
      throws IOException {
    Path path = root.resolve(outputFilename);
    createParentFolder(path);
    try (InputStream is = inputFileProvider.getInputStream(filename);
        OutputStream os = Files.newOutputStream(path)) {
      ByteStreams.copy(is, os);
    }
  }

  @Override
  public void write(String filename, byte[] content) throws IOException {
    Path path = root.resolve(filename);
    createParentFolder(path);
    Files.write(path, content);
  }

  @Override
  public void close() {
    // Nothing to close
  }

  private void createParentFolder(Path path) throws IOException {
    if (!Files.exists(path.getParent())) {
      Files.createDirectories(path.getParent());
    }
  }
}
