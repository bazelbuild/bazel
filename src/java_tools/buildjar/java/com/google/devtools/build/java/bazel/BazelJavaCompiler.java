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

package com.google.devtools.build.java.bazel;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import javax.lang.model.SourceVersion;
import javax.tools.DiagnosticListener;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;

/**
 * Provides a {@link JavaCompiler} that has behavior as similar as possible
 * to the java compiler provided by default by Bazel.
 * Replace calls to {@link javax.tools.ToolProvider#getSystemJavaCompiler}
 * with calls to {@link BazelJavaCompiler#newInstance}.
 *
 * <p>This class is typically used only from a host build tool or in tests.
 * When using this in production, langtools.jar and the bootclasspath jars
 * are deployed as separate jar files within the runfiles directory.
 */
public class BazelJavaCompiler {

  // The default blessed javac options.
  
  private static final String DEFAULT_BOOTCLASSPATH = JavacBootclasspath.asString();

  private static final String[] DEFAULT_JAVACOPTS;
  static {
    List<String> defaultJavacopts = new ArrayList<>();
    for (String javacopt : JavaBuilderConfig.defaultJavacOpts()) {
      if (javacopt.startsWith("-Xep")) {
        // ignore Error Prone-specific flags accepted by JavaBuilder
        continue;
      }
      defaultJavacopts.add(javacopt);
    }

    // The bootclasspath must be specified both via an invocation option and
    // via fileManager.setLocation(PLATFORM_CLASS_PATH), to work around what
    // appears to be a bug in jdk[6,8] javac.
    defaultJavacopts.addAll(Arrays.asList("-bootclasspath", DEFAULT_BOOTCLASSPATH));
    
    DEFAULT_JAVACOPTS = defaultJavacopts.toArray(new String[defaultJavacopts.size()]);
  }

  private static final Class<? extends JavaCompiler> JAVA_COMPILER_CLASS = getJavaCompilerClass();

  private static class LangtoolsClassLoader extends URLClassLoader {

    public LangtoolsClassLoader() throws MalformedURLException {
      super(new URL[] { getLangtoolsJar().toURI().toURL() },
          // We use the bootstrap classloader (null) as the parent classloader
          // instead of the default "system" class loader; we intentionally do
          // not consult the classpath. This way the class path is not
          // polluted, we reduce the risk of having multiple langtools on the
          // classpath, and langtools.jar is only opened if this method is
          // called.  And this will reduce problems for automated java
          // dependency analysis, which other teams are trying to do.
          null);
    }
  }

  private static Class<? extends JavaCompiler> getJavaCompilerClass() {
    try {
      ClassLoader cl = new LangtoolsClassLoader();
      return getJavaCompilerClass(cl);
    } catch (Exception e) {
      throw new RuntimeException("Cannot get java compiler", e);
    }
  }

  private static Class<? extends JavaCompiler> getJavaCompilerClass(ClassLoader cl)
      throws Exception {
    return Class.forName("com.sun.tools.javac.api.JavacTool", true, cl)
        .asSubclass(JavaCompiler.class);
  }

  /**
   * Returns the langtools jar.
   */
  public static File getLangtoolsJar() {
    return JavaLangtools.file();
  }
  
  /**
   * Returns the default javacopts, including the blessed bootclasspath.
   */
  public static List<String> getDefaultJavacopts() {
    return new ArrayList<>(Arrays.asList(DEFAULT_JAVACOPTS));
  }

  /**
   * Returns a new {@link JavaCompiler} that has behavior as similar as
   * possible to the java compiler provided by default by the bazel build
   * system, independent of the user-specified {@code JAVABASE}.
   *
   * <p>More precisely, this method works analogously to {@link
   * javax.tools.ToolProvider#getSystemJavaCompiler}, but returns a {@code
   * JavaCompiler} that differs in these details:
   *
   * <ul>
   *
   * <li> uses the blessed javac implementation: {@code //tools/defaults:java_langtools},
   * as defined by bazel's --java_langtools flag.
   *
   * <li> uses the blessed boot class path: {@code //tools/defaults:javac_bootclasspath},
   * as defined by bazel's --javac_bootclasspath flag.
   *
   * <li> uses the blessed default values for javac options such as {@code -source}
   *
   * </ul>
   *
   * <p>This class ensures that (by default) the {@code -source}, {@code
   * -target} and {@code -bootclasspath} flags all agree and specify the same
   * (blessed) JDK version, for language and API compatibility.
   *
   * <p>This method finds the javac implementation using a custom classloader
   * that does not consult the user's classpath.  This works well, unless the
   * return value is cast to a javac-implementation class like {@code
   * JavacTask}, in which case the dreaded classloader error "can't cast
   * JavacTaskImpl to JavacTask" raises its ugly head, in which case you should
   * use {@link #newInstance(ClassLoader)} instead.
   */
  public static JavaCompiler newInstance() {
    try {
      return newInstance(JAVA_COMPILER_CLASS.getConstructor().newInstance());
    } catch (Exception e) {
      throw new RuntimeException("Cannot get java compiler", e);
    }
  }

  /**
   * Returns a new {@link JavaCompiler} that has behavior as similar as
   * possible to the java compiler provided by default by bazel,
   * independent of the user-specified {@code JAVABASE}.
   *
   * <p>This method has effect identical to {@link #newInstance()} (and that
   * method is generally preferred to this one), except that the javac
   * implementation is found via the provided classloader instead of defining a
   * custom classloader that knows the standard location of the blessed javac
   * implementation.
   *
   * <p>This method is needed when the return value is cast to a
   * javac-implementation class like {@code JavacTask}, to avoid the dreaded
   * multiple classloader error "can't cast JavacTaskImpl to JavacTask".
   *
   * <p>Typically, users should pass {@code ClassLoader.getSystemClassLoader()}
   * as the argument to this method.
   */
  public static JavaCompiler newInstance(ClassLoader cl) {
    try {
      return newInstance(getJavaCompilerClass(cl).getConstructor().newInstance());
    } catch (Exception e) {
      throw new RuntimeException("Cannot get java compiler", e);
    }
  }

  private static JavaCompiler newInstance(final JavaCompiler delegate) {
    // We forward most operations to the JavaCompiler implementation in langtools.jar.
    return new JavaCompiler() {
        @Override
        public CompilationTask getTask(
            Writer out,
            JavaFileManager fileManager,
            DiagnosticListener<? super JavaFileObject> diagnosticListener,
            Iterable<String> options,
            Iterable<String> classes,
            Iterable<? extends JavaFileObject> compilationUnits) {
          // We prepend bazel's default javacopts to user javacopts,
          // so that the user can override them. javac supports this
          // "last option wins" style of option override.
          List<String> fullOptions = getDefaultJavacopts();
          if (options != null) {
            for (String option : options) {
              fullOptions.add(option);
            }
          }
          return delegate.getTask(out,
                                  fileManager,
                                  diagnosticListener,
                                  fullOptions,
                                  classes,
                                  compilationUnits);
        }

        @Override
        public StandardJavaFileManager getStandardFileManager(
            DiagnosticListener<? super JavaFileObject> diagnosticListener,
            Locale locale,
            Charset charset) {
          StandardJavaFileManager fileManager = delegate.getStandardFileManager(
              diagnosticListener,
              locale,
              charset);

          try {
            fileManager.setLocation(
                StandardLocation.PLATFORM_CLASS_PATH,  // bootclasspath
                JavacBootclasspath.asFiles());
          } catch (IOException e) {
            // Should never happen, according to javadocs for setLocation
            throw new RuntimeException(e);
          }
          return fileManager;
        }

        @Override
        public int run(InputStream in, OutputStream out, OutputStream err,
                       String... arguments) {
          // prepend bazel's default javacopts to user arguments
          List<String> args = getDefaultJavacopts();
          args.addAll(Arrays.asList(arguments));
          return delegate.run(in, out, err, args.toArray(new String[0]));
        }

        @Override
        public Set<SourceVersion> getSourceVersions() {
          return delegate.getSourceVersions();
        }

        @Override
        public int isSupportedOption(String option) {
          return delegate.isSupportedOption(option);
        }
      };
  }
}
