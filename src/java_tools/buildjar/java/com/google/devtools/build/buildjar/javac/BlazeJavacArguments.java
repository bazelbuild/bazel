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

package com.google.devtools.build.buildjar.javac;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import java.nio.file.Path;
import javax.annotation.Nullable;
import javax.annotation.processing.Processor;

/**
 * Arguments to a single compilation performed by {@link BlazeJavacMain}.
 *
 * <p>This includes a subset of arguments to {@link javax.tools.JavaCompiler#getTask} and {@link
 * javax.tools.StandardFileManager#setLocation} for a single compilation, with sensible defaults and
 * a builder.
 */
@AutoValue
public abstract class BlazeJavacArguments {
  /** The sources to compile. */
  public abstract ImmutableList<Path> sourceFiles();

  /** Javac options, not including location settings. */
  public abstract ImmutableList<String> javacOptions();

  /** The compilation classpath. */
  public abstract ImmutableList<Path> classPath();

  /** The compilation bootclasspath. */
  public abstract ImmutableList<Path> bootClassPath();

  /** The classpath to load processors from. */
  public abstract ImmutableList<Path> processorPath();

  /**
   * Annotation processor classes. In production builds, processors are specified by string class
   * name in {@link javacOptions}; this is used for tests that instantate processors directly.
   */
  @Nullable
  public abstract ImmutableList<Processor> processors();

  /** The class output directory (-d). */
  public abstract Path classOutput();

  /** The generated source output directory (-s). */
  @Nullable
  public abstract Path sourceOutput();

  public static Builder builder() {
    return new AutoValue_BlazeJavacArguments.Builder()
        .classPath(ImmutableList.of())
        .classOutput(null)
        .bootClassPath(ImmutableList.of())
        .javacOptions(ImmutableList.of())
        .sourceFiles(ImmutableList.of())
        .processors(null)
        .sourceOutput(null)
        .processorPath(ImmutableList.of());
  }

  /** {@link BlazeJavacArguments}Builder. */
  @AutoValue.Builder
  public interface Builder {
    Builder classPath(ImmutableList<Path> classPath);

    Builder classOutput(Path classOutput);

    Builder bootClassPath(ImmutableList<Path> bootClassPath);

    Builder javacOptions(ImmutableList<String> javacOptions);

    Builder sourceFiles(ImmutableList<Path> sourceFiles);

    Builder processors(ImmutableList<Processor> processors);

    Builder sourceOutput(Path sourceOutput);

    Builder processorPath(ImmutableList<Path> processorPath);

    BlazeJavacArguments build();
  }
}
