// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import java.io.IOException;
import java.io.OutputStream;

/**
 * A factory that supplies {@link OutputStream}.
 */
public final class ProvideXmlStreamFactory implements Factory<OutputStream> {
  private final Supplier<JUnit4Config> configSupplier;

  public ProvideXmlStreamFactory(Supplier<JUnit4Config> configSupplier) {
    if (configSupplier == null) {
      throw new IllegalStateException();
    }

    this.configSupplier = configSupplier;
  }

  @Override
  public OutputStream get() {
    OutputStream outputStream =
        new LazyOutputStream(
            new Supplier<OutputStream>() {
              @Override
              public OutputStream get() {
                return JUnit4RunnerModule.provideXmlStream(configSupplier.get());
              }
            });

    return outputStream;
  }

  public static Factory<OutputStream> create(Supplier<JUnit4Config> configSupplier) {
    return new ProvideXmlStreamFactory(configSupplier);
  }

  private static class LazyOutputStream extends OutputStream {
    private Supplier<OutputStream> supplier;
    private volatile OutputStream delegate;

    public LazyOutputStream(Supplier<OutputStream> supplier) {
      this.supplier = supplier;
    }

    private OutputStream ensureDelegate() {
      OutputStream delegate0 = delegate;
      if (delegate0 != null) {
        return delegate0;
      }

      synchronized (this) {
        if (delegate == null) {
          delegate = supplier.get();
          supplier = null;
        }
      }

      return delegate;
    }

    @Override
    public void write(int b) throws IOException {
      ensureDelegate().write(b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      ensureDelegate().write(b, off, len);
    }

    @Override
    public void write(byte[] b) throws IOException {
      ensureDelegate().write(b);
    }

    @Override
    public void close() throws IOException {
      if (delegate != null) {
        delegate.close();
      }
    }

    @Override
    public void flush() throws IOException {
      if (delegate != null) {
        delegate.flush();
      }
    }
  }
}
