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

import com.google.errorprone.annotations.MustBeClosed;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;

/** Input file provider allows to iterate on relative path filename of a directory or a jar file. */
public interface InputFileProvider extends Closeable, Iterable<String> {

  /**
   * Return a ZipEntry for {@code filename}. If the provider is a {@link ZipInputFileProvider}, the
   * method returns the existing ZipEntry in order to keep its metadata, otherwise a new one is
   * created.
   */
  ZipEntry getZipEntry(String filename);

  /**
   * This method returns an input stream allowing to read the file {@code filename}, it is the
   * responsibility of the caller to close this stream.
   */
  InputStream getInputStream(String filename) throws IOException;

  /** Transform a Path to an InputFileProvider that needs to be closed by the caller. */
  @MustBeClosed
  public static InputFileProvider open(Path path) throws IOException {
    if (Files.isDirectory(path)) {
      return new DirectoryInputFileProvider(path);
    } else {
      return new ZipInputFileProvider(path);
    }
  }
}
