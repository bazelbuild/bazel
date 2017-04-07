// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.buildjar;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.javac.BlazeJavacResult;
import com.google.devtools.build.buildjar.javac.FormattedDiagnostic;
import com.google.devtools.build.buildjar.javac.JavacOptions;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule;
import com.google.devtools.build.buildjar.javac.plugins.errorprone.ErrorPronePlugin;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/** The JavaBuilder main called by bazel. */
public abstract class BazelJavaBuilder {

  private static final String CMDNAME = "BazelJavaBuilder";

  /** The main method of the BazelJavaBuilder. */
  public static void main(String[] args) {
    if (args.length == 1 && args[0].equals("--persistent_worker")) {
      System.exit(runPersistentWorker());
    } else {
      // This is a single invocation of JavaBuilder that exits after it processed the request.
      PrintWriter err =
          new PrintWriter(new OutputStreamWriter(System.err, Charset.defaultCharset()));
      int exitCode = processRequest(Arrays.asList(args), err);
      err.flush();
      System.exit(exitCode);
    }
  }

  private static int runPersistentWorker() {
    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

        if (request == null) {
          break;
        }

        try (StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw)) {
          int exitCode = processRequest(request.getArgumentsList(), pw);
          WorkResponse.newBuilder()
              .setOutput(sw.toString())
              .setExitCode(exitCode)
              .build()
              .writeDelimitedTo(System.out);
          System.out.flush();
        }
      } catch (IOException e) {
        e.printStackTrace();
        return 1;
      }
    }
    return 0;
  }

  public static int processRequest(List<String> args, PrintWriter err) {
    try {
      JavaLibraryBuildRequest build = parse(args);
      try (SimpleJavaLibraryBuilder builder =
          build.getDependencyModule().reduceClasspath()
              ? new ReducedClasspathJavaLibraryBuilder()
              : new SimpleJavaLibraryBuilder()) {
        BlazeJavacResult result = builder.run(build);
        for (FormattedDiagnostic d : result.diagnostics()) {
          err.write(d.getFormatted() + "\n");
        }
        err.write(result.output());
        return result.isOk() ? 0 : 1;
      }
    } catch (InvalidCommandLineException e) {
      err.println(CMDNAME + " threw exception: " + e.getMessage());
      return 1;
    } catch (Exception e) {
      e.printStackTrace(err);
      return 1;
    }
  }

  private static boolean processAndRemoveExtraChecksOptions(List<String> args) {
    // error-prone is enabled by default for Bazel.
    boolean errorProneEnabled = true;

    ListIterator<String> arg = args.listIterator();
    while (arg.hasNext()) {
      switch (arg.next()) {
        case "-extra_checks":
        case "-extra_checks:on":
          errorProneEnabled = true;
          arg.remove();
          break;
        case "-extra_checks:off":
          errorProneEnabled = false;
          arg.remove();
          break;
        default: // fall out
      }
    }

    return errorProneEnabled;
  }

  /**
   * Parses the list of arguments into a {@link JavaLibraryBuildRequest}. The returned {@link
   * JavaLibraryBuildRequest} object can be then used to configure the compilation itself.
   *
   * @throws IOException if the argument list contains a file (with the @ prefix) and reading that
   *     file failed
   * @throws InvalidCommandLineException on any command line error
   */
  @VisibleForTesting
  public static JavaLibraryBuildRequest parse(List<String> args)
      throws IOException, InvalidCommandLineException {
    OptionsParser optionsParser = new OptionsParser(args);
    ImmutableList.Builder<BlazeJavaCompilerPlugin> plugins = ImmutableList.builder();

    // Support for -extra_checks:off was removed from ErrorPronePlugin, but Bazel still needs it,
    // so we'll emulate support for this here by handling the flag ourselves and not loading the
    // plug-in when it is specified.
    boolean errorProneEnabled = processAndRemoveExtraChecksOptions(optionsParser.getJavacOpts());
    if (errorProneEnabled) {
      plugins.add(new ErrorPronePlugin());
    }

    JavaLibraryBuildRequest build =
        new JavaLibraryBuildRequest(optionsParser, plugins.build(), new DependencyModule.Builder());
    build.setJavacOpts(JavacOptions.normalizeOptions(build.getJavacOpts()));
    return build;
  }
}
