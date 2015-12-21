// Copyright 2015 The Bazel Authors. All rights reserved.
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

/** Utilities for dealing with {@link Runnable}s that may call into uncontrolled code. */
public class Runnables {
  private Runnables() {}

  /**
   * A {@link Runnable} that terminates the jvm instead of throwing an unchecked exception in
   * {@link Runnable#run}.
   *
   * <p>This is useful if the {@link Runnable} may be executed by code outside of our control, e.g.
   * if we call into library code that silently swallows {@link RuntimeException}s from our code.
   */
  // TODO(bazel-team): Long-term, callers may want to handle exceptions more gracefully.
  public abstract static class AbstractCrashTerminatingRunnable implements Runnable {
    protected abstract void runImpl();

    @Override
    public final void run() {
      try {
        runImpl();
      } catch (Throwable t) {
        RuntimeUtils.halt(t);
      }
    }
  }
}
