// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule;
import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule.StrictJavaDeps;
import com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin;
import com.google.devtools.build.java.turbine.TurbineOptions;
import com.google.devtools.build.java.turbine.TurbineOptionsParser;
import com.google.devtools.build.java.turbine.javac.ZipOutputFileManager.OutputFileObject;

import com.sun.tools.javac.util.Context;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import javax.tools.StandardLocation;

/**
 * An header compiler implementation based on javac.
 *
 * <p>This is a reference implementation used to develop the blaze integration, and to validate
 * the real header compilation implementation.
 */
public class JavacTurbine implements AutoCloseable {

  public static void main(String[] args) throws IOException {
    System.exit(compile(TurbineOptionsParser.parse(Arrays.asList(args))).exitCode());
  }

  public static Result compile(TurbineOptions turbineOptions) throws IOException {
    try (JavacTurbine turbine = new JavacTurbine(turbineOptions)) {
      return turbine.compile();
    }
  }

  /** A header compilation result. */
  enum Result {
    /** The compilation succeeded with the reduced classpath optimization. */
    OK_WITH_REDUCED_CLASSPATH(true),

    /** The compilation succeeded, but had to fall back to a transitive classpath. */
    OK_WITH_FULL_CLASSPATH(true),

    /** The compilation did not succeed. */
    ERROR(false);

    private final boolean ok;

    private Result(boolean ok) {
      this.ok = ok;
    }

    boolean ok() {
      return ok;
    }

    int exitCode() {
      return ok ? 0 : 1;
    }
  }

  private static final int ZIPFILE_BUFFER_SIZE = 1024 * 16;

  private static final Joiner CLASSPATH_JOINER = Joiner.on(':');

  private final PrintWriter out;
  private final TurbineOptions turbineOptions;
  @VisibleForTesting Context context;

  public JavacTurbine(PrintWriter out, TurbineOptions turbineOptions) {
    this.out = out;
    this.turbineOptions = turbineOptions;
  }

  public JavacTurbine(TurbineOptions turbineOptions) {
    this(new PrintWriter(System.err, true), turbineOptions);
  }

  Result compile() throws IOException {
    Path tmpdir = Paths.get(turbineOptions.tempDir());
    Files.createDirectories(tmpdir);

    ImmutableList.Builder<String> argbuilder = ImmutableList.builder();

    addLanguageLevel(argbuilder, turbineOptions.javacOpts());

    // Disable compilation of implicit source files.
    // This is insurance: the sourcepath is empty, so we don't expect implicit sources.
    argbuilder.add("-implicit:none");

    ImmutableList<Path> processorpath;
    if (!turbineOptions.processors().isEmpty()) {
      argbuilder.add("-processor");
      argbuilder.add(Joiner.on(',').join(turbineOptions.processors()));
      processorpath = asPaths(turbineOptions.processorPath());
    } else {
      processorpath = ImmutableList.of();
    }

    List<String> sources = new ArrayList<>();
    sources.addAll(turbineOptions.sources());
    sources.addAll(extractSourceJars(turbineOptions, tmpdir));

    argbuilder.addAll(sources);

    JavacTurbineCompileRequest.Builder requestBuilder =
        JavacTurbineCompileRequest.builder()
            .setJavacOptions(argbuilder.build())
            .setBootClassPath(asPaths(turbineOptions.bootClassPath()))
            .setProcessorClassPath(processorpath);

    StrictJavaDeps strictDepsMode = StrictJavaDeps.valueOf(turbineOptions.strictDepsMode());

    DependencyModule dependencyModule = buildDependencyModule(turbineOptions, strictDepsMode);

    Result result = Result.ERROR;
    JavacTurbineCompileResult compileResult;
    List<String> actualClasspath;

    if (strictDepsMode != StrictJavaDeps.OFF) {

      List<String> originalClasspath = turbineOptions.classPath();
      List<String> compressedClasspath =
          dependencyModule.computeStrictClasspath(turbineOptions.classPath());

      requestBuilder.setStrictDepsPlugin(new StrictJavaDepsPlugin(dependencyModule));

      {
        // compile with reduced classpath
        actualClasspath = compressedClasspath;
        requestBuilder.setClassPath(asPaths(actualClasspath));
        compileResult = JavacTurbineCompiler.compile(requestBuilder.build());
        if (compileResult.success()) {
          result = Result.OK_WITH_REDUCED_CLASSPATH;
          context = compileResult.context();
        }
      }

      if (!compileResult.success() && hasRecognizedError(compileResult.output())) {
        // fall back to transitive classpath
        deleteRecursively(tmpdir);
        extractSourceJars(turbineOptions, tmpdir);

        actualClasspath = originalClasspath;
        requestBuilder.setClassPath(asPaths(actualClasspath));
        compileResult = JavacTurbineCompiler.compile(requestBuilder.build());
        if (compileResult.success()) {
          result = Result.OK_WITH_FULL_CLASSPATH;
          context = compileResult.context();
        }
      }

    } else {
      actualClasspath = turbineOptions.classPath();
      requestBuilder.setClassPath(asPaths(actualClasspath));
      compileResult = JavacTurbineCompiler.compile(requestBuilder.build());
      if (compileResult.success()) {
        result = Result.OK_WITH_FULL_CLASSPATH;
        context = compileResult.context();
      }
    }

    if (!result.ok()) {
      out.println(compileResult.output());
      return result;
    }

    emitClassJar(Paths.get(turbineOptions.outputFile()), compileResult);
    dependencyModule.emitUsedClasspath(CLASSPATH_JOINER.join(actualClasspath));
    dependencyModule.emitDependencyInformation(
        CLASSPATH_JOINER.join(actualClasspath), compileResult.success());

    return result;
  }

  private static DependencyModule buildDependencyModule(
      TurbineOptions turbineOptions, StrictJavaDeps strictDepsMode) {
    DependencyModule.Builder dependencyModuleBuilder =
        new DependencyModule.Builder()
            .setReduceClasspath()
            .setTargetLabel(turbineOptions.targetLabel())
            .addDepsArtifacts(turbineOptions.depsArtifacts())
            .setStrictJavaDeps(strictDepsMode.toString())
            .addDirectMappings(turbineOptions.directJarsToTargets())
            .addIndirectMappings(turbineOptions.indirectJarsToTargets());

    if (turbineOptions.outputDeps().isPresent()) {
      dependencyModuleBuilder.setOutputDepsProtoFile(turbineOptions.outputDeps().get());
    }

    return dependencyModuleBuilder.build();
  }

  /** Write the class output from a successful compilation to the output jar. */
  private static void emitClassJar(Path outputJar, JavacTurbineCompileResult result)
      throws IOException {
    try (OutputStream fos = Files.newOutputStream(outputJar);
        ZipOutputStream zipOut =
            new ZipOutputStream(new BufferedOutputStream(fos, ZIPFILE_BUFFER_SIZE))) {
      boolean hasEntries = false;
      for (Map.Entry<String, OutputFileObject> entry : result.files().entrySet()) {
        if (entry.getValue().location != StandardLocation.CLASS_OUTPUT) {
          continue;
        }
        String name = entry.getKey();
        byte[] bytes = entry.getValue().asBytes();
        if (name.endsWith(".class")) {
          bytes = removeCode(bytes);
        }
        ZipUtil.storeEntry(name, bytes, zipOut);
        hasEntries = true;
      }
      if (!hasEntries) {
        // ZipOutputStream refuses to create a completely empty zip file.
        ZipUtil.storeEntry("dummy", new byte[0], zipOut);
      }
    }
  }

  /**
   * Strip any remaining code attributes.
   *
   * <p>Most code will already have been removed after parsing, but the bytecode will still
   * contain e.g. lowered class and instance initializers.
   */
  // TODO(cushon): add additional stripping to produce ijar-compatible output,
  // e.g. removing private members.
  private static byte[] removeCode(byte[] bytes) {
    ClassWriter cw = new ClassWriter(0);
    new ClassReader(bytes)
        .accept(cw, ClassReader.SKIP_CODE | ClassReader.SKIP_FRAMES | ClassReader.SKIP_DEBUG);
    return cw.toByteArray();
  }

  /** Convert string elements of a classpath to {@link Path}s. */
  private static ImmutableList<Path> asPaths(Iterable<String> classpath) {
    ImmutableList.Builder<Path> result = ImmutableList.builder();
    for (String element : classpath) {
      result.add(Paths.get(element));
    }
    return result.build();
  }

  /** Extract the language level from the javacopts. */
  @VisibleForTesting
  static void addLanguageLevel(
      ImmutableList.Builder<String> javacArgs, Iterable<String> defaultJavacopts) {
    Iterator<String> it = defaultJavacopts.iterator();
    while (it.hasNext()) {
      String curr = it.next();
      switch (curr) {
        case "-source":
        case "-target":
          if (it.hasNext()) {
            javacArgs.add(curr);
            javacArgs.add(it.next());
          }
          break;
        default:
          break;
      }
    }
  }

  /** Extra sources in srcjars to disk. */
  private static List<String> extractSourceJars(TurbineOptions turbineOptions, Path tmpdir)
      throws IOException {
    if (turbineOptions.sourceJars().isEmpty()) {
      return Collections.emptyList();
    }

    ArrayList<String> extractedSources = new ArrayList<>();
    for (String sourceJar : turbineOptions.sourceJars()) {
      try (ZipFile zf = new ZipFile(sourceJar)) {
        Enumeration<? extends ZipEntry> entries = zf.entries();
        while (entries.hasMoreElements()) {
          ZipEntry ze = entries.nextElement();
          if (!ze.getName().endsWith(".java")) {
            continue;
          }
          Path dest = tmpdir.resolve(ze.getName());
          Files.createDirectories(dest.getParent());
          Files.copy(zf.getInputStream(ze), dest);
          extractedSources.add(dest.toAbsolutePath().toString());
        }
      }
    }
    return extractedSources;
  }

  private static final Pattern MISSING_PACKAGE =
      Pattern.compile("error: package ([\\p{javaJavaIdentifierPart}\\.]+) does not exist");

  /**
   * The compilation failed with an error that may indicate that the reduced class path was too
   * aggressive.
   *
   * <p>WARNING: keep in sync with ReducedClasspathJavaLibraryBuilder.
   */
  // TODO(cushon): use a diagnostic listener and match known codes instead
  private static boolean hasRecognizedError(String javacOutput) {
    return javacOutput.contains("error: cannot access")
        || javacOutput.contains("error: cannot find symbol")
        || javacOutput.contains("com.sun.tools.javac.code.Symbol$CompletionFailure")
        || MISSING_PACKAGE.matcher(javacOutput).find();
  }

  @Override
  public void close() throws IOException {
    deleteRecursively(Paths.get(turbineOptions.tempDir()));
  }

  private static void deleteRecursively(final Path dir) throws IOException {
    Files.walkFileTree(
        dir,
        new SimpleFileVisitor<Path>() {
          @Override
          public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
              throws IOException {
            Files.delete(path);
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path path, IOException exc) throws IOException {
            if (!path.equals(dir)) {
              Files.delete(path);
            }
            return FileVisitResult.CONTINUE;
          }
        });
  }
}
