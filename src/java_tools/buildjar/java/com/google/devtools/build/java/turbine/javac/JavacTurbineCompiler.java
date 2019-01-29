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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import com.google.devtools.build.java.turbine.javac.JavacTurbineCompileResult.Status;
import com.sun.source.util.JavacTask;
import com.sun.tools.javac.api.ClientCodeWrapper.Trusted;
import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.file.CacheFSInfo;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.FileSystem;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import javax.annotation.Nullable;
import javax.tools.Diagnostic;
import javax.tools.JavaFileObject;
import javax.tools.StandardLocation;

/** Performs a javac-based turbine compilation given a {@link JavacTurbineCompileRequest}. */
public class JavacTurbineCompiler {

  static JavacTurbineCompileResult compile(JavacTurbineCompileRequest request) throws IOException {

    Map<String, byte[]> files = new LinkedHashMap<>();
    Status status;
    StringWriter sw = new StringWriter();
    ImmutableList.Builder<FormattedDiagnostic> diagnostics = ImmutableList.builder();
    Context context = new Context();

    try (PrintWriter pw = new PrintWriter(sw)) {
      setupContext(context, request.strictJavaDepsPlugin(), request.transitivePlugin());
      CacheFSInfo.preRegister(context);
      try (FileSystem fs = Jimfs.newFileSystem(Configuration.forCurrentPlatform());
          JavacFileManager fm = new ClassloaderMaskingFileManager()) {
        JavacTask task =
            JavacTool.create()
                .getTask(
                    pw,
                    fm,
                    diagnostic -> diagnostics.add(formatDiagnostic(diagnostic)),
                    request.javacOptions(),
                    /* classes= */ ImmutableList.of(),
                    fm.getJavaFileObjectsFromPaths(request.sources()),
                    context);

        Path classes = fs.getPath("classes");
        Files.createDirectories(classes);
        Path sources = fs.getPath("sources");
        Files.createDirectories(sources);

        fm.setContext(context);
        fm.setLocationFromPaths(StandardLocation.SOURCE_PATH, Collections.<Path>emptyList());
        fm.setLocationFromPaths(StandardLocation.CLASS_PATH, request.classPath());
        // The bootclasspath may legitimately be empty if --release is being used.
        Collection<Path> bootClassPath = request.bootClassPath();
        if (!bootClassPath.isEmpty()) {
          fm.setLocationFromPaths(StandardLocation.PLATFORM_CLASS_PATH, bootClassPath);
        }
        fm.setLocationFromPaths(
            StandardLocation.ANNOTATION_PROCESSOR_PATH, request.processorClassPath());
        fm.setLocationFromPaths(StandardLocation.CLASS_OUTPUT, ImmutableList.of(classes));
        fm.setLocationFromPaths(StandardLocation.SOURCE_OUTPUT, ImmutableList.of(sources));

        status = task.call() ? Status.OK : Status.ERROR;

        // collect class output
        Files.walkFileTree(
            classes,
            new SimpleFileVisitor<Path>() {
              @Override
              public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                  throws IOException {
                // use `/` as the directory separator for jar paths, even on Windows
                String name = Joiner.on('/').join(classes.relativize(path));
                files.put(name, Files.readAllBytes(path));
                return FileVisitResult.CONTINUE;
              }
            });

      } catch (Throwable t) {
        t.printStackTrace(pw);
        status = Status.ERROR;
      }
    }

    return new JavacTurbineCompileResult(
        ImmutableMap.copyOf(files), status, sw.toString(), diagnostics.build(), context);
  }

  private static FormattedDiagnostic formatDiagnostic(
      Diagnostic<? extends JavaFileObject> diagnostic) {
    StringBuilder message = new StringBuilder();
    if (diagnostic.getSource() != null) {
      message.append(diagnostic.getSource().getName());
      if (diagnostic.getLineNumber() != -1) {
        message.append(':').append(diagnostic.getLineNumber());
      }
      message.append(": ");
    }
    message.append(diagnostic.getKind().toString().toLowerCase(Locale.getDefault()));
    message.append(": ").append(diagnostic.getMessage(Locale.getDefault()));
    return new FormattedDiagnostic(diagnostic, message.toString());
  }

  /** Mask the annotation processor classpath to avoid version skew. */
  @Trusted
  private static class ClassloaderMaskingFileManager extends JavacFileManager {

    private static Context getContext() {
      Context context = new Context();
      CacheFSInfo.preRegister(context);
      return context;
    }

    public ClassloaderMaskingFileManager() {
      super(getContext(), false, UTF_8);
    }

    @Override
    protected ClassLoader getClassLoader(URL[] urls) {
      return new URLClassLoader(
          urls,
          new ClassLoader(getPlatformClassLoader()) {
            @Override
            protected Class<?> findClass(String name) throws ClassNotFoundException {
              if (name.startsWith("com.sun.source.")
                  || name.startsWith("com.sun.tools.")
                  || name.startsWith("com.google.devtools.build.buildjar.javac.statistics.")) {
                return Class.forName(name);
              }
              throw new ClassNotFoundException(name);
            }
          });
    }
  }

  public static ClassLoader getPlatformClassLoader() {
    try {
      // In JDK 9+, all platform classes are visible to the platform class loader:
      // https://docs.oracle.com/javase/9/docs/api/java/lang/ClassLoader.html#getPlatformClassLoader--
      return (ClassLoader) ClassLoader.class.getMethod("getPlatformClassLoader").invoke(null);
    } catch (ReflectiveOperationException e) {
      // In earlier releases, set 'null' as the parent to delegate to the boot class loader.
      return null;
    }
  }

  static void setupContext(
      Context context, @Nullable StrictJavaDepsPlugin sjd, JavacTransitive transitive) {
    JavacTurbineJavaCompiler.preRegister(context, sjd, transitive);
    BlazeJavacStatistics.preRegister(context);
  }
}
