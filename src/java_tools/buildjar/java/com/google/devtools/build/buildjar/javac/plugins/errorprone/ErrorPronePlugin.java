// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.buildjar.javac.plugins.errorprone;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.errorprone.ErrorProneAnalyzer;
import com.google.errorprone.ErrorProneError;
import com.google.errorprone.ErrorProneOptions;
import com.google.errorprone.InvalidCommandLineOptionException;
import com.google.errorprone.bugpatterns.BugChecker;
import com.google.errorprone.scanner.BuiltInCheckerSuppliers;
import com.google.errorprone.scanner.ScannerSupplier;

import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskEvent.Kind;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.main.Main.Result;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.JavacMessages;
import com.sun.tools.javac.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ServiceLoader;

import javax.tools.JavaFileManager;
import javax.tools.StandardLocation;

/**
 * A plugin for BlazeJavaCompiler that performs Error Prone analysis.
 * Error Prone is a static analysis framework that we use to perform
 * some simple static checks on Java code.
 */
public final class ErrorPronePlugin extends BlazeJavaCompilerPlugin {

  private final Optional<ScannerSupplier> extraChecks;

  public ErrorPronePlugin(ScannerSupplier extraChecks) {
    this.extraChecks = Optional.of(extraChecks);
  }

  public ErrorPronePlugin() {
    this.extraChecks = Optional.absent();
  }

  private ErrorProneAnalyzer errorProneAnalyzer;
  private ErrorProneOptions epOptions;
  // error-prone is enabled by default
  private boolean enabled = true;

  /** Registers our message bundle. */
  public static void setupMessageBundle(Context context) {
    JavacMessages.instance(context).add("com.google.errorprone.errors");
  }

  @Override
  public List<String> processArgs(List<String> args) throws InvalidCommandLineException {
    // allow javacopts that reference unknown error-prone checks
    args = ImmutableList.<String>builder().addAll(args).add("-XepIgnoreUnknownCheckNames").build();
    return processEpOptions(processExtraChecksOption(args));
  }

  private List<String> processEpOptions(List<String> args) throws InvalidCommandLineException {
    try {
      epOptions = ErrorProneOptions.processArgs(args);
    } catch (InvalidCommandLineOptionException e) {
      throw new InvalidCommandLineException(e.getMessage());
    }
    return Arrays.asList(epOptions.getRemainingArgs());
  }

  private List<String> processExtraChecksOption(List<String> args) {
    List<String> arguments = new ArrayList<>();
    for (String arg : args) {
      switch (arg) {
        case "-extra_checks":
        case "-extra_checks:on":
          enabled = true;
          break;
        case "-extra_checks:off":
          enabled = false;
          break;
        default:
          arguments.add(arg);
      }
    }
    return arguments;
  }

  private ScannerSupplier defaultScannerSupplier() {
    // open-source checks that are errors
    ScannerSupplier result = BuiltInCheckerSuppliers.errorChecks();
    if (extraChecks.isPresent()) {
      result = result.plus(extraChecks.get());
    }
    return result;
  }

  private static final Function<BugChecker, Class<? extends BugChecker>> GET_CLASS =
      new Function<BugChecker, Class<? extends BugChecker>>() {
        @Override
        public Class<? extends BugChecker> apply(BugChecker input) {
          return input.getClass();
        }
      };

  @Override
  public void init(Context context, Log log, JavaCompiler compiler) {
    super.init(context, log, compiler);

    if (!enabled) { // error-prone plugin is turned-off
      return;
    }

    setupMessageBundle(context);

    // TODO(cushon): Move this into error-prone proper
    JavaFileManager fileManager = context.get(JavaFileManager.class);
    // Search ANNOTATION_PROCESSOR_PATH if it's available, otherwise fallback to fileManager's
    // own class loader.  Unlike in annotation processor discovery, we never search CLASS_PATH.
    ClassLoader loader = fileManager.hasLocation(StandardLocation.ANNOTATION_PROCESSOR_PATH)
        ? fileManager.getClassLoader(StandardLocation.ANNOTATION_PROCESSOR_PATH)
        : fileManager.getClass().getClassLoader();
    Iterable<BugChecker> extraBugCheckers = ServiceLoader.load(BugChecker.class, loader);
    ScannerSupplier scannerSupplier =
        defaultScannerSupplier().plus(
            ScannerSupplier.fromBugCheckerClasses(
                Iterables.transform(extraBugCheckers, GET_CLASS)));

    if (epOptions != null) {
      try {
        scannerSupplier = scannerSupplier.applyOverrides(epOptions);
      } catch (InvalidCommandLineOptionException e) {
        throwError(Result.CMDERR, e.getMessage());
      }
    } else {
      epOptions = ErrorProneOptions.empty();
    }

    errorProneAnalyzer = ErrorProneAnalyzer.create(scannerSupplier.get()).init(context, epOptions);
  }

  /**
   * Run Error Prone analysis after performing dataflow checks.
   */
  @Override
  public void postFlow(Env<AttrContext> env) {
    if (enabled) {
      try {
        errorProneAnalyzer.finished(new TaskEvent(Kind.ANALYZE, env.toplevel, env.enclClass.sym));
      } catch (ErrorProneError e) {
        e.logFatalError(log);
        // let the exception propagate to javac's main, where it will cause the compilation to
        // terminate with Result.ABNORMAL
        throw e;
      }
    }
  }

  @VisibleForTesting
  public boolean isEnabled() {
    return enabled;
  }
}
