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
package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.InputStream;

/**
 * Input stream that guarantees its contents match a size.
 *
 * <p>This class is not thread safe, but it is safe to message pass this object between threads.
 */
@ThreadCompatible
public class CheckContentLengthInputStream extends InputStream {
  private final InputStream delegate;
  private final long expectedSize;
  private long actualSize;

  public CheckContentLengthInputStream(InputStream delegate, long expectedSize) {
    this.delegate = delegate;
    this.expectedSize = expectedSize;
  }

  @Override
  public int read() throws IOException {
    int result = delegate.read();
    if (result == -1) {
      checkContentLength();
    } else {
      actualSize += 1;
    }
    return result;
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    int amount = delegate.read(buffer, offset, length);
    if (amount == -1) {
      checkContentLength();
    } else {
      actualSize += amount;
    }
    return amount;
  }

  @Override
  public int available() throws IOException {
    return delegate.available();
  }

  @Override
  public void close() throws IOException {
    delegate.close();
    checkContentLength();
  }

  private void checkContentLength() throws IOException {
    if (actualSize != expectedSize) {
      throw new ContentLengthMismatchException(actualSize, expectedSize);
    }
  }
}
