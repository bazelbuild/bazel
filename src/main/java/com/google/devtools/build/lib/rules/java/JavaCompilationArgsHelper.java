// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import java.util.List;

// TODO(cushon): delete this class, see discussion in cl/193146318
@AutoValue
abstract class JavaCompilationArgsHelper {
  abstract boolean recursive();

  abstract boolean isNeverLink();

  abstract boolean srcLessDepsExport();

  abstract JavaCompilationArtifacts compilationArtifacts();

  abstract List<JavaCompilationArgsProvider> depsCompilationArgs();

  abstract List<JavaCompilationArgsProvider> runtimeDepsCompilationArgs();

  abstract List<JavaCompilationArgsProvider> exportsCompilationArgs();

  static Builder builder() {
    return new AutoValue_JavaCompilationArgsHelper.Builder()
        .setDepsCompilationArgs(ImmutableList.of())
        .setRuntimeDepsCompilationArgs(ImmutableList.of())
        .setExportsCompilationArgs(ImmutableList.of());
  }

  public abstract Builder toBuilder();

  @AutoValue.Builder
  abstract static class Builder {
    abstract Builder setRecursive(boolean value);

    abstract Builder setIsNeverLink(boolean value);

    abstract Builder setSrcLessDepsExport(boolean value);

    abstract Builder setCompilationArtifacts(JavaCompilationArtifacts value);

    abstract Builder setDepsCompilationArgs(List<JavaCompilationArgsProvider> value);

    abstract Builder setRuntimeDepsCompilationArgs(List<JavaCompilationArgsProvider> value);

    abstract Builder setExportsCompilationArgs(List<JavaCompilationArgsProvider> value);

    abstract JavaCompilationArgsHelper build();
  }
}
