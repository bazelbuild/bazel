// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.Path;
import java.time.Duration;

/** Environment for running {@link CredentialHelper}s in. */
@AutoValue
public abstract class CredentialHelperEnvironment {
  /** Returns the reporter for reporting events related to {@link CredentialHelper}s. */
  public abstract Reporter getEventReporter();

  /**
   * Returns the (absolute) path to the workspace.
   *
   * <p>Used as working directory when invoking the subprocess.
   */
  public abstract Path getWorkspacePath();

  /**
   * Returns the environment from the Bazel client.
   *
   * <p>Passed as environment variables to the subprocess.
   */
  public abstract ImmutableMap<String, String> getClientEnvironment();

  /** Returns the execution timeout for the helper subprocess. */
  public abstract Duration getHelperExecutionTimeout();

  /** Returns a new builder for {@link CredentialHelperEnvironment}. */
  public static CredentialHelperEnvironment.Builder newBuilder() {
    return new AutoValue_CredentialHelperEnvironment.Builder();
  }

  /** Builder for {@link CredentialHelperEnvironment}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the reporter for reporting events related to {@link CredentialHelper}s. */
    public abstract Builder setEventReporter(Reporter reporter);

    /**
     * Sets the (absolute) path to the workspace to use as working directory when invoking the
     * subprocess.
     */
    public abstract Builder setWorkspacePath(Path path);

    /**
     * Sets the environment from the Bazel client to pass as environment variables to the
     * subprocess.
     */
    public abstract Builder setClientEnvironment(ImmutableMap<String, String> environment);

    /** Sets the execution timeout for the helper subprocess. */
    public abstract Builder setHelperExecutionTimeout(Duration timeout);

    /** Returns the newly constructed {@link CredentialHelperEnvironment}. */
    public abstract CredentialHelperEnvironment build();
  }
}
