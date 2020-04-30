/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.testing.junit;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Iterables;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** The transformation record that describes the desugaring of a jar. */
@AutoValue
abstract class JarTransformationRecord {

  /**
   * The full runtime path of a pre-transformationRecord jar.
   *
   * @see com.google.devtools.build.android.desugar.config.DesugarOptions#inputJars for details.
   */
  abstract ImmutableList<Path> inputJars();

  /**
   * The full runtime path of a post-transformationRecord jar (deguared jar).
   *
   * @see com.google.devtools.build.android.desugar.config.DesugarOptions#inputJars for details.
   */
  abstract ImmutableList<Path> outputJars();

  /** @see com.google.devtools.build.android.desugar.config.DesugarOptions#classpath for details. */
  abstract ImmutableList<Path> classPathEntries();

  /**
   * @see com.google.devtools.build.android.desugar.config.DesugarOptions#bootclasspath for details.
   */
  abstract ImmutableList<Path> bootClassPathEntries();

  /** The remaining command options used for desugaring. */
  abstract ImmutableListMultimap<String, String> extraCustomCommandOptions();

  /** The factory method of this jar transformation record. */
  static JarTransformationRecord create(
      ImmutableList<Path> inputJars,
      ImmutableList<Path> outputJars,
      ImmutableList<Path> classPathEntries,
      ImmutableList<Path> bootClassPathEntries,
      ImmutableListMultimap<String, String> extraCustomCommandOptions) {
    return new AutoValue_JarTransformationRecord(
        inputJars, outputJars, classPathEntries, bootClassPathEntries, extraCustomCommandOptions);
  }

  final ImmutableList<String> getDesugarFlags() {
    ImmutableList.Builder<String> args = ImmutableList.builder();
    inputJars().forEach(path -> args.add("--input=" + path));
    outputJars().forEach(path -> args.add("--output=" + path));
    classPathEntries().forEach(path -> args.add("--classpath_entry=" + path));
    bootClassPathEntries().forEach(path -> args.add("--bootclasspath_entry=" + path));
    extraCustomCommandOptions().forEach((k, v) -> args.add("--" + k + "=" + v));
    return args.build();
  }

  @Memoized
  ClassLoader getOutputClassLoader() throws MalformedURLException {
    List<URL> urls = new ArrayList<>();
    for (Path path : Iterables.concat(outputJars(), classPathEntries(), bootClassPathEntries())) {
      urls.add(path.toUri().toURL());
    }
    return URLClassLoader.newInstance(urls.toArray(new URL[0]), DesugarRule.BASE_CLASS_LOADER);
  }
}
