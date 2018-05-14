// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac.testing;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.auto.value.AutoValue;
import com.google.common.base.CharMatcher;
import com.google.common.base.Joiner;
import com.google.common.base.StandardSystemProperty;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.java.bazel.JavaBuilderConfig;
import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;
import java.io.IOError;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.FileSystem;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaFileObject;
import javax.tools.StandardLocation;

/** A collection of utilities for in-memory compilation testing. */
// TODO(cushon): rename this, it is no longer a file manager
public class InMemoryJavaFileManager {

  protected static final CharMatcher SLASH_MATCHER = CharMatcher.is('/');
  protected static final CharMatcher DOT_MATCHER = CharMatcher.is('.');

  /** The name and bytecode of a compiled class. */
  @AutoValue
  public abstract static class CompiledClass {
    public abstract String name();

    @SuppressWarnings("mutable")
    public abstract byte[] data();

    public static CompiledClass create(String name, byte[] data) {
      return new AutoValue_InMemoryJavaFileManager_CompiledClass(name, data);
    }
  }

  /** Returns the package name for a fully-qualified classname, or the empty string. */
  // TODO(cushon): this doesn't work for nested classes.
  public static String getPackageName(String className) {
    int dot = className.lastIndexOf('.');
    return (dot > 0) ? className.substring(0, dot) : "";
  }

  /** Create a plausible looking target name from a package name, for testing. */
  public static String getTargetName(String pkg) {
    return "//com/google/" + pkg.replace('.', '/');
  }

  /** Builds a collection of sources to compile. */
  public static class SourceBuilder {
    private final Path root;
    private final ImmutableList.Builder<Path> sources = ImmutableList.builder();

    public static SourceBuilder create(FileSystem fileSystem) {
      try {
        Path tmp = fileSystem.getPath("/tmp");
        if (!Files.exists(tmp)) {
          Files.createDirectory(tmp);
        }
        return new SourceBuilder(Files.createTempDirectory(tmp, ""));
      } catch (IOException e) {
        throw new IOError(e);
      }
    }

    public SourceBuilder(Path root) {
      this.root = root;
    }

    public SourceBuilder addSourceLines(String name, String... lines) throws IOException {
      Path path = root.resolve(name);
      Files.createDirectories(path.getParent());
      Files.write(path, Arrays.asList(lines), UTF_8);
      sources.add(path);
      return this;
    }

    public ImmutableList<Path> build() {
      return sources.build();
    }
  }

  /** A wrapper around {@link JavaCompiler}. */
  public static class CompilationBuilder {

    private Context context = new Context();
    private JavacFileManager fileManager = new JavacFileManager(new Context(), false, UTF_8);
    private Collection<Path> sources = Collections.emptyList();
    private Collection<Path> classpath = Collections.emptyList();
    private Iterable<String> javacopts = JavaBuilderConfig.defaultJavacOpts();
    private Path output = null;

    public CompilationBuilder() {}

    public CompilationBuilder setContext(Context context) {
      this.context = context;
      return this;
    }

    public CompilationBuilder setFileManager(JavacFileManager fileManager) {
      this.fileManager = fileManager;
      return this;
    }

    public CompilationBuilder setSources(Collection<Path> sources) {
      this.sources = sources;
      return this;
    }

    public CompilationBuilder setClasspath(Collection<Path> classpath) {
      this.classpath = classpath;
      return this;
    }

    public CompilationBuilder setJavacopts(Iterable<String> javacopts) {
      this.javacopts = javacopts;
      return this;
    }

    public CompilationBuilder setOutput(Path output) {
      this.output = output;
      return this;
    }

    public CompilationResult compile() throws IOException {
      if (output == null) {
        Path tmp =
            sources
                .iterator()
                .next()
                .getFileSystem()
                .getPath(StandardSystemProperty.JAVA_IO_TMPDIR.value());
        if (!Files.exists(tmp)) {
          Files.createDirectory(tmp);
        }
        output = Files.createTempDirectory(tmp, "classes");
        Files.createDirectories(output);
      }
      DiagnosticCollector<JavaFileObject> diagnosticCollector = new DiagnosticCollector<>();
      StringWriter errorOutput = new StringWriter();

      fileManager.setLocationFromPaths(StandardLocation.CLASS_PATH, classpath);
      fileManager.setLocationFromPaths(StandardLocation.SOURCE_PATH, Collections.<Path>emptyList());
      fileManager.setLocationFromPaths(
          StandardLocation.ANNOTATION_PROCESSOR_PATH, Collections.<Path>emptyList());

      fileManager.setLocationFromPaths(
          StandardLocation.CLASS_OUTPUT, Collections.singleton(output));

      boolean ok =
          JavacTool.create()
              .getTask(
                  new PrintWriter(errorOutput, true),
                  fileManager,
                  diagnosticCollector,
                  javacopts,
                  /*classes=*/ Collections.<String>emptyList(),
                  fileManager.getJavaFileObjectsFromPaths(sources),
                  context)
              .call();

      return CompilationResult.create(
          ok, diagnosticCollector.getDiagnostics(), errorOutput.toString(), output);
    }

    public CompilationResult compileOrDie() throws IOException {
      CompilationResult result = compile();
      if (!result.ok()) {
        throw new AssertionError(Joiner.on('\n').join(result.diagnostics()));
      }
      return result;
    }

    public ImmutableList<CompiledClass> toCompiledClassesOrDie() throws IOException {
      return compileOrDie().classData();
    }

    public Path compileOutputToJarOrDie() throws IOException {
      CompilationResult result = compileOrDie();
      return compiledClassesToJar(result.classOutput().resolve("output.jar"), result.classData());
    }
  }

  public static Path compiledClassesToJar(Path jar, Iterable<CompiledClass> classes) {
    return compiledClassesToJar(jar, classes, null);
  }

  public static Path compiledClassesToJar(
      Path jar, Iterable<CompiledClass> classes, String targetLabel) {
    Manifest manifest = new Manifest();
    Attributes attributes = manifest.getMainAttributes();
    attributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");
    if (targetLabel != null) {
      attributes.putValue("Target-Label", targetLabel);
    }
    try (OutputStream os = Files.newOutputStream(jar);
        final JarOutputStream jos = new JarOutputStream(os, manifest)) {
      for (CompiledClass c : classes) {
        jos.putNextEntry(new JarEntry(c.name().replace('.', '/') + ".class"));
        jos.write(c.data());
      }
    } catch (IOException e) {
      throw new IOError(e);
    }
    return jar;
  }

  /** The output from a compilation. */
  @AutoValue
  public abstract static class CompilationResult {
    public abstract boolean ok();

    public abstract ImmutableList<Diagnostic<? extends JavaFileObject>> diagnostics();

    public abstract String errorOutput();

    public abstract Path classOutput();

    public ImmutableList<CompiledClass> classData() throws IOException {
      ImmutableList.Builder<CompiledClass> result = ImmutableList.builder();
      Files.walkFileTree(
          classOutput(),
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                throws IOException {
              if (path.toString().endsWith(".class")) {
                String className = classOutput().relativize(path).toString();
                className = className.substring(0, className.length() - ".class".length());
                className = DOT_MATCHER.trimLeadingFrom(SLASH_MATCHER.replaceFrom(className, '.'));
                result.add(CompiledClass.create(className, Files.readAllBytes(path)));
              }
              return FileVisitResult.CONTINUE;
            }
          });
      return result.build();
    }

    static CompilationResult create(
        boolean ok,
        Iterable<Diagnostic<? extends JavaFileObject>> diagnostics,
        String errorOutput,
        Path classOutput) {
      return new AutoValue_InMemoryJavaFileManager_CompilationResult(
          ok, ImmutableList.copyOf(diagnostics), errorOutput, classOutput);
    }
  }
}
