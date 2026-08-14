// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import java.util.concurrent.atomic.AtomicBoolean;

/** Knobs for memory optimizations. */
public final class MemoryOptimizations {

  private MemoryOptimizations() {}

  /**
   * The value of the --experimental_non_deterministic_memory_optimizations option (on the current
   * command), i.e. whether or not Blaze internals should attempt to do memory optimizations that
   * may be non-deterministic with respect to their efficacy.
   *
   * <p>Set by BlazeCommandDispatcher at the start of each command.
   */
  public static final AtomicBoolean doNonDeterministicMemoryOptimizations = new AtomicBoolean(true);
}
