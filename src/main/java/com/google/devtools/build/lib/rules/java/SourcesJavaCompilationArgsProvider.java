// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * An interface that marks configured targets that can provide Java compilation arguments through
 * the 'srcs' attribute of Java rules.
 *
 * <p>In a perfect world, this would not be necessary for a million reasons, but
 * this world is far from perfect, thus, we need this.
 *
 * <p>Please do not implement this interface with configured target implementations.
 */
@Immutable
public final class SourcesJavaCompilationArgsProvider implements TransitiveInfoProvider {
  private final JavaCompilationArgs javaCompilationArgs;
  private final JavaCompilationArgs recursiveJavaCompilationArgs;

  public SourcesJavaCompilationArgsProvider(
      JavaCompilationArgs javaCompilationArgs,
      JavaCompilationArgs recursiveJavaCompilationArgs) {
    this.javaCompilationArgs = javaCompilationArgs;
    this.recursiveJavaCompilationArgs = recursiveJavaCompilationArgs;
  }

  /**
   * Returns non-recursively collected Java compilation information for
   * building this target (called when strict_java_deps = 1).
   *
   * <p>Note that some of the parameters are still collected from the complete
   * transitive closure. The non-recursive collection applies mainly to
   * compile-time jars.
   */
  public JavaCompilationArgs getJavaCompilationArgs() {
    return javaCompilationArgs;
  }

  /**
   * Returns recursively collected Java compilation information for building
   * this target (called when strict_java_deps = 0).
   */
  public JavaCompilationArgs getRecursiveJavaCompilationArgs() {
    return recursiveJavaCompilationArgs;
  }
}
