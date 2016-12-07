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
package com.google.devtools.build.lib.util;

/**
 * A {@link ThreadSafeBatchCallback} that trivially delegates to a {@link BatchCallback} in a
 * synchronized manner.
 */
public class SynchronizedBatchCallback<T, E extends Exception>
    implements ThreadSafeBatchCallback<T, E> {
  private final BatchCallback<T, E> delegate;

  public SynchronizedBatchCallback(BatchCallback<T, E> delegate) {
    this.delegate = delegate;
  }

  @Override
  public synchronized void process(Iterable<T> partialResult) throws E, InterruptedException {
    delegate.process(partialResult);
  }
}

