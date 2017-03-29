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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * The result of a spawn execution.
 */
public interface SpawnResult {
  /**
   * Returns whether the spawn was actually run, regardless of the exit code. Returns false if there
   * were network errors, missing local files, errors setting up sandboxing, etc.
   */
  boolean setupSuccess();

  /**
   * The exit code of the subprocess.
   */
  int exitCode();

  /**
   * Basic implementation of {@link SpawnResult}.
   */
  @Immutable @ThreadSafe
  public static final class SimpleSpawnResult implements SpawnResult {
    private final boolean setupSuccess;
    private final int exitCode;

    SimpleSpawnResult(Builder builder) {
      this.setupSuccess = builder.setupSuccess;
      this.exitCode = builder.exitCode;
    }

    @Override
    public boolean setupSuccess() {
      return setupSuccess;
    }

    @Override
    public int exitCode() {
      return exitCode;
    }
  }

  /**
   * Builder class for {@link SpawnResult}.
   */
  public static final class Builder {
    private boolean setupSuccess;
    private int exitCode;

    public SpawnResult build() {
      return new SimpleSpawnResult(this);
    }

    public Builder setSetupSuccess(boolean setupSuccess) {
      this.setupSuccess = setupSuccess;
      return this;
    }

    public Builder setExitCode(int exitCode) {
      this.exitCode = exitCode;
      return this;
    }
  }
}
