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
package com.google.devtools.build.lib.query2.engine;

import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A {@link ThreadSafeOutputFormatterCallback} wrapper around a {@link OutputFormatterCallback}
 * delegate.
 */
public final class SynchronizedDelegatingOutputFormatterCallback<T>
    extends ThreadSafeOutputFormatterCallback<T> {
  private final OutputFormatterCallback<T> delegate;

  public SynchronizedDelegatingOutputFormatterCallback(OutputFormatterCallback<T> delegate) {
    this.delegate = delegate;
  }

  @Override
  public synchronized void start() throws IOException  {
    delegate.start();
  }

  @Override
  public synchronized void close(boolean failFast) throws InterruptedException, IOException {
    delegate.close(failFast);
  }

  @Override
  public synchronized void process(Iterable<T> partialResult)
      throws QueryException, InterruptedException {
    delegate.process(partialResult);
  }

  @Override
  public synchronized void processOutput(Iterable<T> partialResult)
      throws IOException, InterruptedException {
    delegate.processOutput(partialResult);
  }

  @Override
  @Nullable
  public IOException getIoException() {
    return delegate.getIoException();
  }
}
