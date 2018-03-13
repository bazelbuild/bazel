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

import com.google.common.base.Functions;
import com.google.common.collect.Iterators;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** Input provider is a zip file. */
class ZipInputFileProvider implements InputFileProvider {

  private final Path root;

  private final ZipFile zipFile;

  public ZipInputFileProvider(Path root) throws IOException {
    this.root = root;
    this.zipFile = new ZipFile(root.toFile());
  }

  @Override
  public void close() throws IOException {
    zipFile.close();
  }

  @Override
  public String toString() {
    return root.getFileName().toString();
  }

  @Override
  public ZipEntry getZipEntry(String filename) {
    ZipEntry zipEntry = zipFile.getEntry(filename);
    zipEntry.setCompressedSize(-1);
    return zipEntry;
  }

  @Override
  public InputStream getInputStream(String filename) throws IOException {
    return zipFile.getInputStream(zipFile.getEntry(filename));
  }

  @Override
  public Iterator<String> iterator() {
    return Iterators.transform(
        Iterators.forEnumeration(zipFile.entries()), Functions.toStringFunction());
  }
}
