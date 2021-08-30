package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.InputStream;

/**
 * Input stream that guarantees its contents matches a size.
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
      check();
    } else {
      actualSize += result;
    }
    return result;
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    int amount = delegate.read(buffer, offset, length);
    if (amount == -1) {
      check();
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
    check();
  }

  private void check() throws IOException {
    if (expectedSize != actualSize) {
      throw new UnrecoverableHttpException(
          String.format(
              "Connection was closed before Content-Length was fulfilled. Bytes read %s but wanted %s",
              actualSize, expectedSize));
    }
  }
}
