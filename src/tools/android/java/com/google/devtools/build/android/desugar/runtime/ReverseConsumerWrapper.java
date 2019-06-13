// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.runtime;

import java.util.function.Consumer;

/** Conversion from built-in to desugared {@link Consumer}s (b/134636762). */
@SuppressWarnings("AndroidJdkLibsChecker")
public final class ReverseConsumerWrapper<T> implements j$.util.function.Consumer<T> {
  public static <T> j$.util.function.Consumer<T> fromConsumer(Consumer<T> consumer) {
    if (consumer == null) {
      return null;
    }
    return new ReverseConsumerWrapper<>(consumer);
  }

  private final Consumer<T> wrapped;

  private ReverseConsumerWrapper(Consumer<T> wrapped) {
    this.wrapped = wrapped;
  }

  @Override
  public void accept(T t) {
    wrapped.accept(t);
  }

  @Override
  public j$.util.function.Consumer<T> andThen(j$.util.function.Consumer<? super T> after) {
    // TODO(b/134636762): Support chaining consumers
    throw new UnsupportedOperationException("Not supported on wrapped consumers");
  }
}
