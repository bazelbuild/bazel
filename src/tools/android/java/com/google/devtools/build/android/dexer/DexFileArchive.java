// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import com.android.dex.Dex;
import com.android.dx.dex.file.DexFile;
import com.google.common.io.ByteStreams;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Wrapper around a {@link ZipOutputStream} to simplify writing archives with {@code .dex} files.
 * Adding files generally requires a {@link ZipEntry} in order to control timestamps.
 */
class DexFileArchive implements Closeable {

  private final ZipOutputStream out;

  public DexFileArchive(ZipOutputStream out) {
    this.out = out;
  }

  /**
   * Copies the content of the given {@link InputStream} into an entry with the given details.
   */
  public DexFileArchive copy(ZipEntry entry, InputStream in) throws IOException {
    out.putNextEntry(entry);
    ByteStreams.copy(in, out);
    out.closeEntry();
    return this;
  }

  /**
   * Serializes and adds a {@code .dex} file with the given details.
   */
  public DexFileArchive addFile(ZipEntry entry, DexFile dex) throws IOException {
    return addFile(entry, DexFiles.encode(dex));
  }

  /**
   * Adds a {@code .dex} file with the given details.
   */
  public DexFileArchive addFile(ZipEntry entry, Dex dex) throws IOException {
    entry.setSize(dex.getLength());
    out.putNextEntry(entry);
    dex.writeTo(out);
    out.closeEntry();
    return this;
  }

  @Override
  public void close() throws IOException {
    out.close();
  }

  private DexFileArchive addFile(ZipEntry entry, byte[] content) throws IOException {
    entry.setSize(content.length);
    out.putNextEntry(entry);
    out.write(content);
    out.closeEntry();
    return this;
  }
}
