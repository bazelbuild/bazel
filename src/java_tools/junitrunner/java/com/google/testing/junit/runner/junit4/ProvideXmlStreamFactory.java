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
    assert configSupplier != null;
    this.configSupplier = configSupplier;
  }

  @Override
  public OutputStream get() {
    OutputStream outputStream = new LazyOutputStream(
            () -> JUnit4RunnerModule.provideXmlStream(configSupplier.get()));
    return outputStream;
  }

  public static Factory<OutputStream> create(Supplier<JUnit4Config> configSupplier) {
    return new ProvideXmlStreamFactory(configSupplier);
  }

  private static class LazyOutputStream extends OutputStream {
    private Supplier<OutputStream> supplier;
    private OutputStream underlying;

    public LazyOutputStream(Supplier<OutputStream> supplier) {
      this.supplier = supplier;
    }

    public void write(int b) throws IOException {
      if (underlying == null) {
        synchronized (this) {
          if (underlying == null) {
            underlying = supplier.get();
            supplier = null;
          }
        }
      }

      underlying.write(b);
    }

    @Override
    public void close() throws IOException {
      if (underlying != null) {
        underlying.close();
      }
    }

    @Override
    public void flush() throws IOException {
      if (underlying != null) {
        underlying.flush();
      }
    }
  }
}
