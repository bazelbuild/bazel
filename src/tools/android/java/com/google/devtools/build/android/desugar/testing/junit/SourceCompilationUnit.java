/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.tools.JavaCompiler;

/** Represents a compilation unit with a single jar output. */
public final class SourceCompilationUnit {

  /** The java compiler used to compile source files. */
  private final JavaCompiler compiler;

  /**
   * customJavacOptions The javac options used for compilation, with the same support of `javacopts`
   * attribute in java_binary rule.
   */
  private final ImmutableList<String> customJavacOptions;

  /** The collection of source files subject to compile. */
  private final ImmutableList<Path> sourceInputs;

  /** The caller-specified write-permissible file path to the compiled jar. */
  private final Path outputJar;

  public SourceCompilationUnit(
      JavaCompiler compiler,
      ImmutableList<String> customJavacOptions,
      ImmutableList<Path> sourceInputs,
      Path outputJar) {
    this.compiler = compiler;
    this.customJavacOptions = customJavacOptions;
    this.sourceInputs = sourceInputs;
    this.outputJar = outputJar;
  }

  /** Compiles Java source files and write to the pre-specified path to the output jar. */
  Path compile() throws IOException, SourceCompilationException {
    Path compilationStdOut = Files.createTempFile("compilation_stdout_", ".txt");
    Path compilationStdErr = Files.createTempFile("compilation_stderr_", ".txt");
    Path compiledRootDir = Files.createTempDirectory("compilation_prodout_");
    ImmutableList<String> javacOptions =
        ImmutableList.<String>builder()
            .addAll(customJavacOptions)
            .add("-d " + compiledRootDir)
            .build();
    final List<Path> compiledFiles;
    try (OutputStream stdOutStream = Files.newOutputStream(compilationStdOut);
        OutputStream stdErrStream = Files.newOutputStream(compilationStdErr)) {
      Splitter splitter = Splitter.on(" ").trimResults().omitEmptyStrings();
      ImmutableList<String> compilationArguments =
          ImmutableList.<String>builder()
              .addAll(splitter.split(String.join(" ", javacOptions)))
              .addAll(sourceInputs.stream().map(Path::toString).collect(Collectors.toList()))
              .build();
      compiler.run(
          nullInputStream(),
          stdOutStream,
          stdErrStream,
          compilationArguments.toArray(new String[0]));
      int maxDepth = sourceInputs.stream().mapToInt(Path::getNameCount).max().getAsInt();
      try (Stream<Path> outputStream =
          Files.find(compiledRootDir, maxDepth, (path, fileAttr) -> true)) {
        compiledFiles = outputStream.collect(Collectors.toList());
      }
      try (JarOutputStream jarOutputStream =
          new JarOutputStream(Files.newOutputStream(outputJar))) {
        for (Path compiledFile : compiledFiles) {
          try (InputStream inputStream = Files.newInputStream(compiledFile)) {
            Path inArchivalPath = compiledRootDir.relativize(compiledFile);
            JarEntry jarEntry = new JarEntry(inArchivalPath.toString());
            jarOutputStream.putNextEntry(jarEntry);
            if (!Files.isDirectory(compiledFile)) {
              ByteStreams.copy(inputStream, jarOutputStream);
            }
            jarOutputStream.closeEntry();
          }
        }
      }
    }
    String compilationStandardErrorMessage =
        new String(Files.readAllBytes(compilationStdErr), Charset.defaultCharset());
    if (!compilationStandardErrorMessage.isEmpty()) {
      throw new SourceCompilationException(compilationStandardErrorMessage);
    }
    return outputJar;
  }

  private static InputStream nullInputStream() {
    return new ByteArrayInputStream(new byte[] {});
  }

  static class SourceCompilationException extends Exception {
    public SourceCompilationException(String message) {
      super(message);
    }
  }
}
