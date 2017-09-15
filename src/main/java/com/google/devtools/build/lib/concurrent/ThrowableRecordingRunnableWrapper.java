// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import com.google.devtools.build.lib.util.Preconditions;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * A class that wraps Runnables and records the first Throwable thrown by the wrapped Runnables
 * when they are run.
 */
public class ThrowableRecordingRunnableWrapper {

  private final String name;
  private AtomicReference<Throwable> errorRef = new AtomicReference<>();

  private static final Logger logger =
      Logger.getLogger(ThrowableRecordingRunnableWrapper.class.getName());

  public ThrowableRecordingRunnableWrapper(String name) {
    this.name = Preconditions.checkNotNull(name);
  }

  @Nullable
  public Throwable getFirstThrownError() {
    return errorRef.get();
  }

  public Runnable wrap(final Runnable runnable) {
    return () -> {
      try {
        runnable.run();
      } catch (Throwable error) {
        errorRef.compareAndSet(null, error);
        logger.log(Level.SEVERE, "Error thrown by runnable in " + name, error);
      }
    };
  }
}
