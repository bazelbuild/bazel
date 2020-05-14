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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.io.ByteStreams;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/** Output provider is a zip file. */
class ZipOutputFileProvider implements OutputFileProvider {

  private final ZipOutputStream out;

  public ZipOutputFileProvider(Path root) throws IOException {
    out = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(root)));
  }

  @Override
  public void copyFrom(String filename, InputFileProvider inputFileProvider, String outputFilename)
      throws IOException {
    if (filename.equals(outputFilename)) {
      copyFrom(filename, inputFileProvider, inputFileProvider.getZipEntry(filename));
    } else {
      ZipEntry inputEntry = inputFileProvider.getZipEntry(filename);
      // Clone inputEntry with new name, since there's no ZipEntry.setName()
      ZipEntry result = new ZipEntry(outputFilename);
      result.setTime(inputEntry.getTime());
      result.setCrc(inputEntry.getCrc());
      result.setSize(inputEntry.getSize());
      result.setCompressedSize(inputEntry.getCompressedSize());
      result.setMethod(inputEntry.getMethod());
      copyFrom(filename, inputFileProvider, result);
    }
  }

  private void copyFrom(String filename, InputFileProvider inputFileProvider, ZipEntry outputEntry)
      throws IOException {
    // TODO(bazel-team): Avoid de- and re-compressing resource files
    out.putNextEntry(outputEntry);
    try (InputStream is = inputFileProvider.getInputStream(filename)) {
      ByteStreams.copy(is, out);
    }
    out.closeEntry();
  }

  @Override
  public void write(String filename, byte[] content) throws IOException {
    checkArgument(
        filename.equals(DESUGAR_DEPS_FILENAME) || filename.endsWith(".class"),
        "Expect file to be copied: %s",
        filename);
    writeStoredEntry(out, filename, content);
  }

  @Override
  public void close() throws IOException {
    out.close();
  }

  private static void writeStoredEntry(ZipOutputStream out, String filename, byte[] content)
      throws IOException {
    // Need to pre-compute checksum for STORED (uncompressed) entries)
    CRC32 checksum = new CRC32();
    checksum.update(content);

    ZipEntry result = new ZipEntry(filename);
    result.setTime(0L); // Use stable timestamp Jan 1 1980
    result.setCrc(checksum.getValue());
    result.setSize(content.length);
    result.setCompressedSize(content.length);
    // Write uncompressed, since this is just an intermediary artifact that
    // we will convert to .dex
    result.setMethod(ZipEntry.STORED);

    out.putNextEntry(result);
    out.write(content);
    out.closeEntry();
  }
}
