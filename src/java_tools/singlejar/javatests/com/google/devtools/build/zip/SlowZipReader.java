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
package com.google.devtools.build.zip;

import static com.google.common.base.Preconditions.checkArgument;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

/**
 * A test ZIP file reader that provides slow {@link InputStream}s for reading sections of the file.
 *
 * <p>Providing InputStreams that do not read or skip the requested amount allows testing that the
 * parsing code properly handles cases where the data is not immediately available.
 */
public class SlowZipReader extends ZipReader {

  /**
   * Opens a zip file for raw acceess. Passes through to {@link ZipReader#ZipReader(File)}.
   */
  public SlowZipReader(File file) throws IOException {
    super(file);
  }

  /**
   * Opens a zip file for raw acceess. Passes through to {@link ZipReader#ZipReader(File, Charset)}.
   */
  public SlowZipReader(File file, Charset charset) throws IOException {
    super(file, charset);
  }

  /**
   * Opens a zip file for raw acceess. Passes through to
   * {@link ZipReader#ZipReader(File, Charset, boolean)}.
   */
  public SlowZipReader(File file, Charset charset, boolean strictEntries) throws IOException {
    super(file, charset, strictEntries);
  }

  /**
   * Returns a new slow {@link InputStream} positioned at fileOffset.
   *
   * @throws IOException if an I/O error has occurred
   */
  @Override
  protected InputStream getStreamAt(long fileOffset) throws IOException {
    final InputStream in = super.getStreamAt(fileOffset);
    return new InputStream() {
      @Override
      public int read() throws IOException {
        return in.read();
      }

      @Override
      public int read(byte[] b, int off, int len) throws IOException {
        checkArgument(b != null);
        checkArgument((len >= 0) && (off >= 0));
        checkArgument(len <= b.length - off);
        if (len == 0) {
          return 0;
        }
        int value = read();
        if (value == -1) {
          return -1;
        }
        b[off] = (byte) value;
        return 1;
      }

      @Override
      public long skip(long n) throws IOException {
        return super.skip(1);
      }
    };
  }
}
