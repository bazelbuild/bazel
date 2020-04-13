// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.OptionsParser.ReduceClasspathMode;
import com.google.devtools.build.buildjar.javac.BlazeJavacResult;
import com.google.devtools.build.buildjar.javac.FormattedDiagnostic;
import com.google.devtools.build.buildjar.javac.JavacRunner;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import java.io.IOException;
import java.nio.file.Path;

/**
 * A variant of SimpleJavaLibraryBuilder that attempts to reduce the compile-time classpath right
 * before invoking the compiler, based on extra information from provided .jdeps files. This mode is
 * enabled via the --reduce_classpath flag, only when Blaze runs with --experimental_java_classpath.
 *
 * <p>A fall-back mechanism detects whether javac fails because the classpath is incorrectly
 * discarding required entries, and re-attempts to compile with the full classpath.
 */
public class ReducedClasspathJavaLibraryBuilder extends SimpleJavaLibraryBuilder {

  /**
   * Attempts to minimize the compile-time classpath before invoking javac, falling back to a
   * regular compile.
   *
   * @param build A JavaLibraryBuildRequest request object describing what to compile
   * @throws IOException clean-up up the output directory fails
   */
  @Override
  BlazeJavacResult compileSources(JavaLibraryBuildRequest build, JavacRunner javacRunner)
      throws IOException {
    // Minimize classpath, but only if we're actually compiling some sources (some invocations of
    // JavaBuilder are only building resource jars).
    ImmutableList<Path> compressedClasspath = build.getClassPath();
    if (!build.getSourceFiles().isEmpty()
        && build.reduceClasspathMode() == ReduceClasspathMode.JAVABUILDER_REDUCED) {
      compressedClasspath =
          build.getDependencyModule().computeStrictClasspath(build.getClassPath());
    }

    // Compile!
    BlazeJavacResult result =
        javacRunner.invokeJavac(build.toBlazeJavacArguments(compressedClasspath));

    // If javac errored out because of missing entries on the classpath, give it another try.
    // TODO(b/119712048): check performance impact of additional retries.
    boolean fallback = shouldFallBack(build, result);
    if (fallback) {
      if (build.reduceClasspathMode() == ReduceClasspathMode.BAZEL_REDUCED) {
        return BlazeJavacResult.fallback();
      }
      if (build.reduceClasspathMode() == ReduceClasspathMode.JAVABUILDER_REDUCED) {
        result = fallback(build, javacRunner);
      }
    }

    BlazeJavacStatistics.Builder stats =
        result.statistics().toBuilder()
            .minClasspathLength(build.getDependencyModule().getImplicitDependenciesMap().size());
    build.getProcessors().stream()
        .map(p -> p.substring(p.lastIndexOf('.') + 1))
        .forEachOrdered(stats::addProcessor);

    switch (build.reduceClasspathMode()) {
      case BAZEL_REDUCED:
      case BAZEL_FALLBACK:
        stats.transitiveClasspathLength(build.fullClasspathLength());
        stats.reducedClasspathLength(build.reducedClasspathLength());
        stats.transitiveClasspathFallback(
            build.reduceClasspathMode() == ReduceClasspathMode.BAZEL_FALLBACK);
        break;
      case JAVABUILDER_REDUCED:
        stats.transitiveClasspathLength(build.getClassPath().size());
        stats.reducedClasspathLength(compressedClasspath.size());
        stats.transitiveClasspathFallback(fallback);
        break;
      default:
        throw new AssertionError(build.reduceClasspathMode());
    }
    return result.withStatistics(stats.build());
  }

  private BlazeJavacResult fallback(JavaLibraryBuildRequest build, JavacRunner javacRunner)
      throws IOException {
    // TODO(cushon): warn for transitive classpath fallback

    // Reset output directories
    prepareSourceCompilation(build);

    // Fall back to the regular compile, but add extra checks to catch transitive uses
    return javacRunner.invokeJavac(build.toBlazeJavacArguments(build.getClassPath()));
  }

  private static boolean shouldFallBack(JavaLibraryBuildRequest build, BlazeJavacResult result) {
    if (result.isOk()) {
      return false;
    }
    if (result.diagnostics().stream().anyMatch(FormattedDiagnostic::maybeReducedClasspathError)) {
      return true;
    }
    if (result.output().contains("com.sun.tools.javac.code.Symbol$CompletionFailure")) {
      return true;
    }
    if (!build.getProcessors().isEmpty()) {
      // If annotation processing is enabled, we have no idea whether or not reduced classpaths are
      // implicated in any errors we see, so always fall back on failing builds.
      return true;
    }
    return false;
  }
}
