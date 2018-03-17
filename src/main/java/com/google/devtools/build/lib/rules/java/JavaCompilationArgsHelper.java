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
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import java.util.List;

@AutoValue
abstract class JavaCompilationArgsHelper {
  abstract boolean recursive();

  abstract boolean isNeverLink();

  abstract boolean srcLessDepsExport();

  abstract JavaCompilationArtifacts compilationArtifacts();

  abstract List<JavaCompilationArgsProvider> depsCompilationArgs();

  abstract Iterable<? extends TransitiveInfoCollection> deps();

  abstract List<JavaCompilationArgsProvider> runtimeDepsCompilationArgs();

  abstract Iterable<? extends TransitiveInfoCollection> runtimeDeps();

  abstract List<JavaCompilationArgsProvider> exportsCompilationArgs();

  abstract Iterable<? extends TransitiveInfoCollection> exports();

  static Builder builder() {
    return new AutoValue_JavaCompilationArgsHelper.Builder()
        .setDepsCompilationArgs(ImmutableList.of())
        .setDeps(ImmutableList.of())
        .setRuntimeDepsCompilationArgs(ImmutableList.of())
        .setRuntimeDeps(ImmutableList.of())
        .setExportsCompilationArgs(ImmutableList.of())
        .setExports(ImmutableList.of());
  }

  public abstract Builder toBuilder();

  @AutoValue.Builder
  abstract static class Builder {
    abstract Builder setRecursive(boolean value);

    abstract Builder setIsNeverLink(boolean value);

    abstract Builder setSrcLessDepsExport(boolean value);

    abstract Builder setCompilationArtifacts(JavaCompilationArtifacts value);

    abstract Builder setDepsCompilationArgs(List<JavaCompilationArgsProvider> value);

    abstract Builder setDeps(Iterable<? extends TransitiveInfoCollection> value);

    abstract Builder setRuntimeDepsCompilationArgs(List<JavaCompilationArgsProvider> value);

    abstract Builder setRuntimeDeps(Iterable<? extends TransitiveInfoCollection> value);

    abstract Builder setExportsCompilationArgs(List<JavaCompilationArgsProvider> value);

    abstract Builder setExports(Iterable<? extends TransitiveInfoCollection> value);

    abstract JavaCompilationArgsHelper build();
  }
}
