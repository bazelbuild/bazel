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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.protobuf.ByteString;
import java.nio.file.Path;
import java.util.OptionalInt;
import javax.annotation.Nullable;

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

  /** Blaze-specific Javac options. */
  public abstract ImmutableList<String> blazeJavacOptions();

  /** The compilation classpath. */
  public abstract ImmutableList<Path> classPath();

  /** The compilation bootclasspath. */
  public abstract ImmutableList<Path> bootClassPath();

  @Nullable
  public abstract Path system();

  /** The compilation source path. */
  public abstract ImmutableList<Path> sourcePath();

  public abstract ImmutableSet<String> builtinProcessors();

  /** The classpath to load processors from. */
  public abstract ImmutableList<Path> processorPath();

  /** The compiler plugins. */
  public abstract ImmutableList<BlazeJavaCompilerPlugin> plugins();

  /** The class output directory (-d). */
  public abstract Path classOutput();

  /** The native header output directory (-h). */
  @Nullable
  public abstract Path nativeHeaderOutput();

  /** The generated source output directory (-s). */
  @Nullable
  public abstract Path sourceOutput();

  /** Stop compiling after the first diagnostic that could cause transitive classpath fallback. */
  public abstract boolean failFast();

  /** The Inputs' path and digest received from a WorkRequest */
  public abstract ImmutableMap<String, ByteString> inputsAndDigest();

  public abstract OptionalInt requestId();

  public static Builder builder() {
    return new AutoValue_BlazeJavacArguments.Builder()
        .classPath(ImmutableList.of())
        .bootClassPath(ImmutableList.of())
        .javacOptions(ImmutableList.of())
        .blazeJavacOptions(ImmutableList.of())
        .sourceFiles(ImmutableList.of())
        .sourcePath(ImmutableList.of())
        .sourceOutput(null)
        .builtinProcessors(ImmutableSet.of())
        .processorPath(ImmutableList.of())
        .plugins(ImmutableList.of())
        .failFast(false)
        .inputsAndDigest(ImmutableMap.of())
        .requestId(OptionalInt.empty());
  }

  /** {@link BlazeJavacArguments}Builder. */
  @AutoValue.Builder
  public interface Builder {
    Builder classPath(ImmutableList<Path> classPath);

    Builder classOutput(Path classOutput);

    Builder nativeHeaderOutput(Path nativeHeaderOutput);

    Builder bootClassPath(ImmutableList<Path> bootClassPath);

    Builder system(Path system);

    Builder javacOptions(ImmutableList<String> javacOptions);

    Builder blazeJavacOptions(ImmutableList<String> javacOptions);

    Builder sourcePath(ImmutableList<Path> sourcePath);

    Builder sourceFiles(ImmutableList<Path> sourceFiles);

    Builder builtinProcessors(ImmutableSet<String> builtinProcessors);

    Builder sourceOutput(Path sourceOutput);

    Builder processorPath(ImmutableList<Path> processorPath);

    Builder plugins(ImmutableList<BlazeJavaCompilerPlugin> plugins);

    Builder failFast(boolean failFast);

    Builder inputsAndDigest(ImmutableMap<String, ByteString> inputsAndDigest);

    Builder requestId(OptionalInt requestId);

    BlazeJavacArguments build();
  }
}
