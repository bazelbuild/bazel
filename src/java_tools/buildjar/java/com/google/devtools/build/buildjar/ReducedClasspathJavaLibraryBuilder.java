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

import com.google.devtools.build.buildjar.javac.JavacRunner;

import com.sun.tools.javac.main.Main.Result;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.regex.Pattern;

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
   * @return result code of the javac compilation
   * @throws IOException clean-up up the output directory fails
   */
  @Override
  Result compileSources(JavaLibraryBuildRequest build, JavacRunner javacRunner, PrintWriter err)
      throws IOException {
    // Minimize classpath, but only if we're actually compiling some sources (some invocations of
    // JavaBuilder are only building resource jars).
    String compressedClasspath = build.getClassPath();
    if (!build.getSourceFiles().isEmpty()) {
      compressedClasspath =
          build.getDependencyModule().computeStrictClasspath(build.getClassPath());
    }
    String[] javacArguments = makeJavacArguments(build, compressedClasspath);

    // Compile!
    StringWriter javacOutput = new StringWriter();
    PrintWriter javacOutputWriter = new PrintWriter(javacOutput);
    Result result = javacRunner.invokeJavac(javacArguments, javacOutputWriter);
    javacOutputWriter.close();

    // If javac errored out because of missing entries on the classpath, give it another try.
    // TODO(bazel-team): check performance impact of additional retries.
    if (!result.isOK() && hasRecognizedError(javacOutput.toString())) {
      if (debug) {
        err.println("warning: [transitive] Target uses transitive classpath to compile.");
      }

      // Reset output directories
      prepareSourceCompilation(build);

      // Fall back to the regular compile, but add extra checks to catch transitive uses
      javacArguments = makeJavacArguments(build);
      result = javacRunner.invokeJavac(javacArguments, err);
    } else {
      err.print(javacOutput.getBuffer());
    }
    return result;
  }

  private static final Pattern MISSING_PACKAGE =
      Pattern.compile("error: package ([\\p{javaJavaIdentifierPart}\\.]+) does not exist");

  private boolean hasRecognizedError(String javacOutput) {
    return javacOutput.contains("error: cannot access")
        || javacOutput.contains("error: cannot find symbol")
        || javacOutput.contains("com.sun.tools.javac.code.Symbol$CompletionFailure")
        || MISSING_PACKAGE.matcher(javacOutput).find();
  }
}
