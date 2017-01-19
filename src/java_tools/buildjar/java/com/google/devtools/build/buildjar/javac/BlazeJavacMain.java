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

package com.google.devtools.build.buildjar.javac;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.FormattedDiagnostic.Listener;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin.PluginException;
import com.sun.source.util.JavacTask;
import com.sun.tools.javac.api.ClientCodeWrapper.Trusted;
import com.sun.tools.javac.api.JavacTaskImpl;
import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.main.Main.Result;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.PropagatedException;
import java.io.IOError;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.List;
import javax.tools.StandardLocation;

/**
 * Main class for our custom patched javac.
 *
 * <p>This main class tweaks the standard javac log class by changing the compiler's context to use
 * our custom log class. This custom log class modifies javac's output to list all errors after all
 * warnings.
 */
public class BlazeJavacMain {

  /**
   * Sets up a BlazeJavaCompiler with the given plugins within the given context.
   *
   * @param context JavaCompiler's associated Context
   */
  @VisibleForTesting
  static void setupBlazeJavaCompiler(
      ImmutableList<BlazeJavaCompilerPlugin> plugins, Context context) {
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      plugin.initializeContext(context);
    }
    BlazeJavaCompiler.preRegister(context, plugins);
  }

  public static BlazeJavacResult compile(BlazeJavacArguments arguments) {

    List<String> javacArguments = arguments.javacOptions();
    try {
      javacArguments = processPluginArgs(arguments.plugins(), javacArguments);
    } catch (InvalidCommandLineException e) {
      return BlazeJavacResult.error(e.getMessage());
    }

    Context context = new Context();
    setupBlazeJavaCompiler(arguments.plugins(), context);

    Result result = Result.ABNORMAL;
    StringWriter errOutput = new StringWriter();
    // TODO(cushon): where is this used when a diagnostic listener is registered? Consider removing
    // it and handling exceptions directly in callers.
    PrintWriter errWriter = new PrintWriter(errOutput);
    Listener diagnostics = new Listener(context);
    BlazeJavaCompiler compiler;

    JavacFileManager fileManager = new ClassloaderMaskingFileManager();
    JavacTask task =
        JavacTool.create()
            .getTask(
                errWriter,
                fileManager,
                diagnostics,
                javacArguments,
                ImmutableList.of() /*classes*/,
                fileManager.getJavaFileObjectsFromPaths(arguments.sourceFiles()),
                context);
    if (arguments.processors() != null) {
      task.setProcessors(arguments.processors());
    }
    fileManager.setContext(context);
    setLocations(fileManager, arguments);
    try {
      try {
        result = ((JavacTaskImpl) task).doCall();
      } catch (PropagatedException e) {
        throw e.getCause();
      }
    } catch (PluginException e) {
      errWriter.println(e.getMessage());
      result = e.getResult();
    } catch (Throwable t) {
      t.printStackTrace(errWriter);
      result = Result.ABNORMAL;
    } finally {
      compiler = (BlazeJavaCompiler) JavaCompiler.instance(context);
      if (result.isOK()) {
        // There could be situations where we incorrectly skip Error Prone and the compilation
        // ends up succeeding, e.g., if there are errors that are fixed by subsequent round of
        // annotation processing.  This check ensures that if there were any flow events at all,
        // then plugins were run.  There may legitimately not be any flow events, e.g. -proc:only
        // or empty source files.
        if (compiler.skippedFlowEvents() > 0 && compiler.flowEvents() == 0) {
          errWriter.println("Expected at least one FLOW event");
          result = Result.ABNORMAL;
        }
      }
    }
    errWriter.flush();
    return new BlazeJavacResult(result, diagnostics.build(), errOutput.toString(), compiler);
  }

  /** Processes Plugin-specific arguments and removes them from the args array. */
  @VisibleForTesting
  static List<String> processPluginArgs(
      ImmutableList<BlazeJavaCompilerPlugin> plugins, List<String> args)
      throws InvalidCommandLineException {
    List<String> processedArgs = args;
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      processedArgs = plugin.processArgs(processedArgs);
    }
    return processedArgs;
  }

  private static void setLocations(JavacFileManager fileManager, BlazeJavacArguments arguments) {
    try {
      fileManager.setLocationFromPaths(StandardLocation.CLASS_PATH, arguments.classPath());
      fileManager.setLocationFromPaths(
          StandardLocation.CLASS_OUTPUT, ImmutableList.of(arguments.classOutput()));
      fileManager.setLocationFromPaths(StandardLocation.SOURCE_PATH, ImmutableList.of());
      // TODO(cushon): require an explicit bootclasspath
      Iterable<Path> bootClassPath = arguments.bootClassPath();
      if (!Iterables.isEmpty(bootClassPath)) {
        fileManager.setLocationFromPaths(StandardLocation.PLATFORM_CLASS_PATH, bootClassPath);
      }
      fileManager.setLocationFromPaths(
          StandardLocation.ANNOTATION_PROCESSOR_PATH, arguments.processorPath());
      if (arguments.sourceOutput() != null) {
        fileManager.setLocationFromPaths(
            StandardLocation.SOURCE_OUTPUT, ImmutableList.of(arguments.sourceOutput()));
      }
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  /**
   * When Bazel invokes JavaBuilder, it puts javac.jar on the bootstrap class path and
   * JavaBuilder_deploy.jar on the user class path. We need Error Prone to be available on the
   * annotation processor path, but we want to mask out any other classes to minimize class version
   * skew.
   */
  @Trusted
  private static class ClassloaderMaskingFileManager extends JavacFileManager {

    public ClassloaderMaskingFileManager() {
      super(new Context(), false, UTF_8);
    }

    @Override
    protected ClassLoader getClassLoader(URL[] urls) {
      return new URLClassLoader(
          urls,
          new ClassLoader(JavacFileManager.class.getClassLoader()) {
            @Override
            protected Class<?> findClass(String name) throws ClassNotFoundException {
              if (name.startsWith("com.google.errorprone.")) {
                return Class.forName(name);
              } else if (name.startsWith("org.checkerframework.dataflow.")) {
                return Class.forName(name);
              } else {
                throw new ClassNotFoundException(name);
              }
            }
          });
    }
  }

  private BlazeJavacMain() {}
}
