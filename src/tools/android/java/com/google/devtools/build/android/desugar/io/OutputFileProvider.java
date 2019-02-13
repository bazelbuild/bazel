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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Output file provider allows to write files in directory or jar files. */
public interface OutputFileProvider extends AutoCloseable {

  /** Filename to use to write out dependency metadata for later consistency checking. */
  public static final String DESUGAR_DEPS_FILENAME = "META-INF/desugar_deps";

  /**
   * Copy {@code filename} from {@code inputFileProvider} to this output, possibly with new name. If
   * input file provider is a {@link ZipInputFileProvider} then the metadata of the zip entry are
   * kept.
   */
  void copyFrom(String filename, InputFileProvider inputFileProvider, String outputFilename)
      throws IOException;

  /** Write {@code content} in {@code filename} to this output */
  void write(String filename, byte[] content) throws IOException;

  /** Transform a Path to an {@link OutputFileProvider} */
  @MustBeClosed
  public static OutputFileProvider create(Path path) throws IOException {
    if (Files.isDirectory(path)) {
      return new DirectoryOutputFileProvider(path);
    } else {
      return new ZipOutputFileProvider(path);
    }
  }
}
