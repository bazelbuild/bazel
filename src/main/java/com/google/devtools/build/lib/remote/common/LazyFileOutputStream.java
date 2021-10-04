// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Creates an {@link OutputStream} that isn't actually opened until the first data is written. This
 * is useful to only have as many open file descriptors as necessary at a time to avoid running into
 * system limits.
 */
public class LazyFileOutputStream extends OutputStream {

  private final Path path;
  private OutputStream out;

  public LazyFileOutputStream(Path path) {
    this.path = path;
  }

  @Override
  public void write(byte[] b) throws IOException {
    ensureOpen();
    out.write(b);
  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    ensureOpen();
    out.write(b, off, len);
  }

  @Override
  public void write(int b) throws IOException {
    ensureOpen();
    out.write(b);
  }

  @Override
  public void flush() throws IOException {
    ensureOpen();
    out.flush();
  }

  @Override
  public void close() throws IOException {
    ensureOpen();
    out.close();
  }

  private void ensureOpen() throws IOException {
    if (out == null) {
      out = path.getOutputStream();
    }
  }
}
