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
import static java.util.Locale.ENGLISH;

import com.google.auto.value.AutoValue;
import com.google.common.base.CharMatcher;
import com.google.common.collect.ImmutableList;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;

import com.sun.tools.javac.nio.JavacPathFileManager;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;

import java.io.IOError;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.file.FileSystem;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import javax.tools.JavaFileObject;
import javax.tools.StandardLocation;

/**
 * An in memory file manager based on {@link JavacPathFileManager} and Jimfs, with utilities for
 * creating Java source files and manipulating compiled classes.
 */
@com.sun.tools.javac.api.ClientCodeWrapper.Trusted
public class InMemoryJavaFileManager extends JavacPathFileManager {

  protected static final CharMatcher SLASH_MATCHER = CharMatcher.is('/');
  protected static final CharMatcher DOT_MATCHER = CharMatcher.is('.');

  private FileSystem fileSystem;
  private final List<Path> sources = new ArrayList<>();

  // Upstream may eventually create a {@code JavaCompiler#getPathFileManager()} that we could
  // use instead. (See implementation comment on {@link PathFileManager}.))
  private static Context makeContext() {
    Context context = new Context();
    context.put(Locale.class, ENGLISH);
    context.put(Log.outKey, new PrintWriter(new OutputStreamWriter(System.err, UTF_8), true));
    return context;
  }

  public InMemoryJavaFileManager() {
    super(makeContext(), false, UTF_8);
    this.fileSystem = Jimfs.newFileSystem(Configuration.unix());
    setDefaultFileSystem(fileSystem);
  }

  /**
   * Creates an in-memory Jar from all compiled classes from package pkg.
   *
   * @param pkg name of package
   * @param compiled a list of classes and their compiled bytecode to include in the jar
   * @return the name of the constructed jar
   */
  public Path makeJarInClasspath(String pkg, List<CompiledClass> compiled) {
    try {
      String jarName = "lib" + pkg.replace(".", "") + ".jar";
      Path jarPath = fileSystem.getPath("/" + jarName);
      try (ZipOutputStream out = new ZipOutputStream(Files.newOutputStream(jarPath))) {
        for (CompiledClass compiledClass : compiled) {
          if (pkg.equals(getPackageName(compiledClass.name()))) {
            String entry = DOT_MATCHER.replaceFrom(compiledClass.name(), '/') + ".class";
            out.putNextEntry(new ZipEntry(entry));
            out.write(compiledClass.data());
          }
        }
      }
      return jarPath;
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  /**
   * Creates an in-memory Java source file with the specified name and content.
   *
   * @param className the fully-qualified class name
   * @param lines Java source code
   */
  public void addSource(String className, String... lines) {
    Path path = fileSystem.getPath(
        "/", SLASH_MATCHER.trimLeadingFrom(DOT_MATCHER.replaceFrom(className, '/') + ".java"));
    try {
      Files.createDirectories(path.getParent());
      Files.write(path, Arrays.asList(lines), UTF_8);
    } catch (IOException e) {
      throw new IOError(e);
    }
    sources.add(path);
  }

  /**
   * Gets a list of available Java sources and clears the fileManager's cache.
   *
   * @return a list of the JavaFileObjects holding the source files
   */
  public List<JavaFileObject> takeAvailableSources() {
    try {
      return ImmutableList.copyOf(getJavaFileObjectsFromPaths(sources));
    } finally {
      sources.clear();
    }
  }

  /**
   * Copy the given JavaFileObjects into the current file manager's filesystem.
   *
   * <p>The previous implementation allowed {@link JavaFileObject}s to be passed between file
   * managers.
   */
  // TODO(cushon): consider making this class return a wrapper around the content of source files
  // other than JavaFileObjects, and then materialize them later.
  public List<JavaFileObject> naturalize(
      JavacPathFileManager external, Iterable<JavaFileObject> externalFileObjects) {
    ImmutableList.Builder<JavaFileObject> result = ImmutableList.builder();
    for (JavaFileObject externalFileObject : externalFileObjects) {
      try {
        Path externalPath = external.getPath(externalFileObject);
        Path dest = fileSystem.getPath(externalPath.toString());
        Files.createDirectories(dest.getParent());
        Files.copy(external.getPath(externalFileObject), dest);
        result.addAll(getJavaFileObjects(dest));
      } catch (IOException e) {
        throw new IOError(e);
      }
    }
    return result.build();
  }

  /**
   * The name and bytecode of a compiled class.
   */
  @AutoValue
  public abstract static class CompiledClass {
    public abstract String name();
    @SuppressWarnings("mutable")
    public abstract byte[] data();

    public static CompiledClass create(String name, byte[] data) {
      return new AutoValue_InMemoryJavaFileManager_CompiledClass(name, data);
    }
  }

  /**
   * Return the list of all compiled classes in the filesystem.
   */
  public List<CompiledClass> getCompiledClasses() {
    final ImmutableList.Builder<CompiledClass> result = ImmutableList.builder();
    try {
      Files.walkFileTree(fileSystem.getPath("/"), new SimpleFileVisitor<Path>() {
        @Override
        public FileVisitResult visitFile(Path filePath, BasicFileAttributes attrs)
            throws IOException {
          if (filePath.toString().endsWith(".class")) {
            String className = filePath.toString();
            className = className.substring(0, className.length() - ".class".length());
            className = DOT_MATCHER.trimLeadingFrom(SLASH_MATCHER.replaceFrom(className, '.'));
            result.add(CompiledClass.create(className, Files.readAllBytes(filePath)));
          }
          return FileVisitResult.CONTINUE;
        }
      });
    } catch (IOException e) {
      throw new IOError(e);
    }
    return result.build();
  }

  /**
   * Returns the package name for a fully-qualified classname, or the empty string.
   */
  // TODO(cushon): this doesn't work for nested classes.
  public static String getPackageName(String className) {
    int dot = className.lastIndexOf('.');
    return (dot > 0) ? className.substring(0, dot) : "";
  }

  /**
   * Create a plausible looking target name from a package name, for testing.
   */
  public static String getTargetName(String pkg) {
    return "//com/google/" + pkg.replace('.', '/');
  }

  /**
   * Set the filemanager's classpath for a compilation. Also explicitly sets the sourcepath and
   * processorpath to empty, since the default behaviour searches for processors and additional
   * sources to compile.
   *
   * <p>We don't rely on the command-line flags, since filemanager flag parsing doesn't work when
   * using javac through the API.
   */
  public void initializeClasspath(Iterable<Path> classpath) {
    try {
      setLocation(StandardLocation.CLASS_PATH, classpath);
      setLocation(StandardLocation.SOURCE_PATH, ImmutableList.<Path>of());
      setLocation(StandardLocation.ANNOTATION_PROCESSOR_PATH, ImmutableList.<Path>of());
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  /**
   * Set the filemanager's bootclasspath for a compilation.
   *
   * <p>We don't rely on the command-line flags, since filemanager flag parsing doesn't work when
   * using javac through the API.
   */
  public void initializeBootClasspath(Iterable<Path> bootClasspath) {
    try {
      setLocation(StandardLocation.PLATFORM_CLASS_PATH, bootClasspath);
    } catch (IOException e) {
      throw new IOError(e);
    }
  }
}
