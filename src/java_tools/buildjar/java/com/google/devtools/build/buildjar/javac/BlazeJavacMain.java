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

import static com.google.common.base.Verify.verifyNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin.PluginException;

import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskListener;
import com.sun.tools.javac.api.JavacTaskImpl;
import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.api.MultiTaskListener;
import com.sun.tools.javac.main.Main;
import com.sun.tools.javac.main.Main.Result;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Options;
import com.sun.tools.javac.util.PropagatedException;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import javax.annotation.processing.Processor;
import javax.tools.DiagnosticListener;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;

/**
 * Main class for our custom patched javac.
 *
 * <p> This main class tweaks the standard javac log class by changing the
 * compiler's context to use our custom log class. This custom log class
 * modifies javac's output to list all errors after all warnings.
 */
public class BlazeJavacMain {

  /**
   * Compose {@link com.sun.tools.javac.main.Main} and perform custom setup before deferring to
   * its compile() method.
   *
   * <p>Historically BlazeJavacMain extended javac's Main and overrode methods to get the desired
   * custom behaviour. That approach created incompatibilities when upgrading to newer versions of
   * javac, so composition is preferred.
   */
  private List<BlazeJavaCompilerPlugin> plugins;
  private final PrintWriter errOutput;
  private final String compilerName;
  private BlazeJavaCompiler compiler = null;

  public BlazeJavacMain(PrintWriter errOutput, List<BlazeJavaCompilerPlugin> plugins) {
    this.compilerName = "blaze javac";
    this.errOutput = errOutput;
    this.plugins = plugins;
  }

  /**
   * Installs the BlazeJavaCompiler within the provided context. Enables
   * plugins based on field values.
   *
   * @param context JavaCompiler's associated Context
   */
  void setupBlazeJavaCompiler(Context context) {
    preRegister(context, plugins);
  }

  private final Function<BlazeJavaCompiler, Void> compilerListener =
      new Function<BlazeJavaCompiler, Void>() {
        @Override
        public Void apply(BlazeJavaCompiler compiler) {
          Verify.verify(BlazeJavacMain.this.compiler == null);
          BlazeJavacMain.this.compiler = Preconditions.checkNotNull(compiler);
          return null;
        }
      };

  public void preRegister(Context context, List<BlazeJavaCompilerPlugin> plugins) {
    this.plugins = plugins;
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      plugin.initializeContext(context);
    }
    BlazeJavaCompiler.preRegister(context, plugins, compilerListener);
  }

  public Result compile(String[] argv) {
    // set up a fresh Context with our custom bindings for JavaCompiler
    Context context = new Context();

    Options options = Options.instance(context);

    // enable Java 8-style type inference features
    //
    // This is currently duplicated in JAVABUILDER. That's deliberate for now, because
    // (1) JavaBuilder's integration test coverage for default options isn't very good, and
    // (2) the migration from JAVABUILDER to java_toolchain configs is in progress so blaze
    // integration tests for defaults options are also not trustworthy.
    //
    // TODO(bazel-team): removed duplication with JAVABUILDER
    options.put("usePolyAttribution", "true");
    options.put("useStrictMethodClashCheck", "true");
    options.put("useStructuralMostSpecificResolution", "true");
    options.put("useGraphInference", "true");

    String[] processedArgs;

    try {
      processedArgs = processPluginArgs(argv);
    } catch (InvalidCommandLineException e) {
      errOutput.println(e.getMessage());
      return Result.CMDERR;
    }

    setupBlazeJavaCompiler(context);
    return compile(processedArgs, context);
  }

  @VisibleForTesting
  public Result compile(String[] argv, Context context) {
    enableEndPositions(context);
    Result result = Result.ABNORMAL;
    try {
      result = new Main(compilerName, errOutput).compile(argv, context);
    } catch (PropagatedException e) {
      if (e.getCause() instanceof PluginException) {
        PluginException pluginException = (PluginException) e.getCause();
        errOutput.println(pluginException.getMessage());
        return pluginException.getResult();
      }
      e.printStackTrace(errOutput);
      result = Result.ABNORMAL;
    } finally {
      if (result.isOK()) {
        verifyNotNull(compiler);
        // There could be situations where we incorrectly skip Error Prone and the compilation
        // ends up succeeding, e.g., if there are errors that are fixed by subsequent round of
        // annotation processing.  This check ensures that if there were any flow events at all,
        // then plugins were run.  There may legitimately not be any flow events, e.g. -proc:only
        // or empty source files.
        if (compiler.skippedFlowEvents() > 0 && compiler.flowEvents() == 0) {
          errOutput.println("Expected at least one FLOW event");
          result = Result.ABNORMAL;
        }
      }
    }
    return result;
  }

  // javac9 removes the ability to pass lists of {@link JavaFileObject}s or {@link Processors}s to
  // it's 'Main' class (i.e. the entry point for command-line javac). Having BlazeJavacMain
  // continue to call into javac's Main has the nice property that it keeps JavaBuilder's
  // behaviour closer to stock javac, but it makes it harder to write integration tests. This class
  // provides a compile method that accepts file objects and processors, but it isn't
  // guaranteed to behave exactly the same way as JavaBuilder does when used from the command-line.
  // TODO(cushon): either stop using Main and commit to using the the API for everything, or
  // re-write integration tests for JavaBuilder to use the real compile() method.
  @VisibleForTesting
  @Deprecated
  public Result compile(
      String[] argv,
      Context context,
      JavaFileManager fileManager,
      DiagnosticListener<? super JavaFileObject> diagnosticListener,
      List<JavaFileObject> javaFileObjects,
      Iterable<? extends Processor> processors) {

    JavacTool tool = JavacTool.create();
    JavacTaskImpl task = (JavacTaskImpl) tool.getTask(
        errOutput,
        fileManager,
        diagnosticListener,
        Arrays.asList(argv),
        null,
        javaFileObjects,
        context);
    if (processors != null) {
      task.setProcessors(processors);
    }

    try {
      return task.doCall();
    } catch (PluginException e) {
      errOutput.println(e.getMessage());
      return e.getResult();
    }
  }

  private static final TaskListener EMPTY_LISTENER = new TaskListener() {
    @Override public void started(TaskEvent e) {}
    @Override public void finished(TaskEvent e) {}
  };

  /**
   * Convinces javac to run in 'API mode', and collect end position information needed by
   * error-prone.
   */
  private static void enableEndPositions(Context context) {
    MultiTaskListener.instance(context).add(EMPTY_LISTENER);
  }

  /**
   * Processes Plugin-specific arguments and removes them from the args array.
   */
  @VisibleForTesting
  String[] processPluginArgs(String[] args) throws InvalidCommandLineException {
    List<String> processedArgs = Arrays.asList(args);
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      processedArgs = plugin.processArgs(processedArgs);
    }
    return processedArgs.toArray(new String[processedArgs.size()]);
  }

  @VisibleForTesting
  BlazeJavaCompiler getCompiler() {
    return verifyNotNull(compiler);
  }
}
