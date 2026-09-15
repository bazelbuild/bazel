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

package com.google.devtools.build.lib.rules.java;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import javax.annotation.Nullable;

/** The outputs of a {@link JavaCompileAction}. */
@AutoValue
public abstract class JavaCompileOutputs<T extends Artifact> {

  /** The class jar Artifact to create with the Action */
  public abstract T output();

  /** The output artifact for the manifest proto emitted from JavaBuilder */
  public abstract T manifestProto();

  @Nullable
  public abstract T depsProto();

  /** The generated class jar, or {@code null} if no annotation processing is expected. */
  @Nullable
  public abstract T genClass();

  /**
   * The generated sources jar Artifact to create with the Action (null if no sources will be
   * generated).
   */
  @Nullable
  public abstract T genSource();

  /** An archive of generated native header files. */
  @Nullable
  public abstract T nativeHeader();

  static <T extends Artifact> Builder<T> builder() {
    return new AutoValue_JavaCompileOutputs.Builder<>();
  }

  public abstract Builder<T> toBuilder();

  public JavaCompileOutputs<T> withOutput(T output) {
    return toBuilder().output(output).build();
  }

  @AutoValue.Builder
  abstract static class Builder<T extends Artifact> {

    abstract Builder<T> output(T artifact);

    abstract Builder<T> manifestProto(T artifact);

    abstract Builder<T> depsProto(@Nullable T artifact);

    abstract Builder<T> genClass(@Nullable T artifact);

    abstract Builder<T> genSource(@Nullable T artifact);

    abstract Builder<T> nativeHeader(@Nullable T artifact);

    abstract JavaCompileOutputs<T> build();
  }
}
