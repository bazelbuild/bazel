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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;

/** Input provider is a directory. */
class DirectoryInputFileProvider implements InputFileProvider {

  private final Path root;

  public DirectoryInputFileProvider(Path root) {
    this.root = root;
  }

  @Override
  public String toString() {
    return root.getFileName().toString();
  }

  @Override
  public InputStream getInputStream(String filename) throws IOException {
    return Files.newInputStream(root.resolve(filename));
  }

  @Override
  public ZipEntry getZipEntry(String filename) {
    ZipEntry destEntry = new ZipEntry(filename);
    destEntry.setTime(0L); // Use stable timestamp Jan 1 1980
    return destEntry;
  }

  @Override
  public void close() throws IOException {
    // Nothing to close
  }

  @Override
  public Iterator<String> iterator() {
    final List<String> entries = new ArrayList<>();
    try (Stream<Path> paths = Files.walk(root)) {
      paths.forEach(
          new Consumer<Path>() {
            @Override
            public void accept(Path t) {
              if (Files.isRegularFile(t)) {
                // Internally, we use '/' as a common package separator in filename to abstract
                // that filename can comes from a zip or a directory.
                entries.add(root.relativize(t).toString().replace(File.separatorChar, '/'));
              }
            }
          });
    } catch (IOException e) {
      throw new IOError(e);
    }
    return entries.iterator();
  }
}
