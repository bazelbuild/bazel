// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.skyframe.SkyFunction.Environment;

/**
 * Encapsulates {@link Environment#dependOnFuture}, {@link Environment#valuesMissing} in a single
 * method.
 *
 * <p>Decoupling this from {@link Environment} simplifies testing and the API.
 */
public interface DependOnFutureShim {
  /** Returned status of {@link #dependOnFuture}. */
  enum ObservedFutureStatus {
    /** If the future was already done. */
    DONE,
    /**
     * If the future was not done.
     *
     * <p>Indicates that a Skyframe restart is needed.
     */
    NOT_DONE
  }

  /**
   * Outside of testing, delegates to {@link Environment#dependOnFuture} then {@link
   * Environment#valuesMissing} to determine the return value.
   *
   * @throws IllegalStateException if called when an underlying environment's {@link
   *     Environment#valuesMissing} is already true.
   */
  ObservedFutureStatus dependOnFuture(ListenableFuture<?> future);

  /** A convenience implementation used with an {@link Environment} instance. */
  final class DefaultDependOnFutureShim implements DependOnFutureShim {
    private final Environment env;

    public DefaultDependOnFutureShim(Environment env) {
      this.env = env;
    }

    @Override
    public ObservedFutureStatus dependOnFuture(ListenableFuture<?> future) {
      checkState(!env.valuesMissing());
      env.dependOnFuture(future);
      return env.valuesMissing() ? ObservedFutureStatus.NOT_DONE : ObservedFutureStatus.DONE;
    }
  }
}
