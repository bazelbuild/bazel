// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.framework;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** {@link ProcessRunner} parameters */
@AutoValue
public abstract class ProcessParameters {
  abstract String name();

  abstract ImmutableList<String> arguments();

  abstract File workingDirectory();

  abstract int expectedExitCode();

  abstract boolean expectedToFail();

  abstract boolean expectedEmptyError();

  abstract Optional<ImmutableMap<String, String>> environment();

  abstract long timeoutMillis();

  abstract Optional<Path> redirectOutput();

  abstract Optional<Path> redirectError();

  public static Builder builder() {
    return new AutoValue_ProcessParameters.Builder()
        .setExpectedExitCode(0)
        .setExpectedEmptyError(true)
        .setExpectedToFail(false)
        .setTimeoutMillis(30 * 1000)
        .setArguments();
  }

  /** Builder class */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setName(String value);

    public abstract Builder setArguments(String... args);

    public abstract Builder setArguments(ImmutableList<String> args);

    public Builder setArguments(List<String> args) {
      setArguments(ImmutableList.copyOf(args));
      return this;
    }

    public abstract Builder setWorkingDirectory(File value);

    public abstract Builder setExpectedExitCode(int value);

    public abstract Builder setExpectedToFail(boolean value);

    public abstract Builder setExpectedEmptyError(boolean value);

    public abstract Builder setEnvironment(ImmutableMap<String, String> map);

    public Builder setEnvironment(Map<String, String> map) {
      setEnvironment(ImmutableMap.copyOf(map));
      return this;
    }

    public abstract Builder setTimeoutMillis(long millis);

    public abstract Builder setRedirectOutput(Path path);

    public abstract Builder setRedirectError(Path path);

    public abstract ProcessParameters build();
  }
}
