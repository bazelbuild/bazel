// Copyright 2007 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.buildjar.javac.JavacOptions;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.classloader.ClassLoaderMaskingPlugin;
import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule;
import com.google.devtools.build.buildjar.javac.plugins.errorprone.ErrorPronePlugin;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;

/**
 * The JavaBuilder main called by bazel.
 */
public abstract class BazelJavaBuilder {

  private static final String CMDNAME = "BazelJavaBuilder";

  /**
   * The main method of the BazelJavaBuilder.
   */
  public static void main(String[] args) {
    if (args.length == 1 && args[0].equals("--persistent_worker")) {
      System.exit(runPersistentWorker());
    } else {
      // This is a single invocation of JavaBuilder that exits after it processed the request.
      System.exit(processRequest(Arrays.asList(args)));
    }
  }

  private static int runPersistentWorker() {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

        if (request == null) {
          break;
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos, true);
        // Make sure that we exit nonzero in case an exception occurs during processRequest.
        int exitCode = 1;
        // TODO(philwo) - change this so that a PrintWriter can be passed in and will be used
        // instead of redirect stdout / stderr.
        System.setOut(ps);
        System.setErr(ps);
        try {
          exitCode = processRequest(request.getArgumentsList());
        } finally {
          System.setOut(originalStdOut);
          System.setErr(originalStdErr);
        }

        WorkResponse.newBuilder()
            .setOutput(baos.toString())
            .setExitCode(exitCode)
            .build()
            .writeDelimitedTo(System.out);
        System.out.flush();
      } catch (IOException e) {
        e.printStackTrace();
        return 1;
      }
    }
    return 0;
  }

  private static int processRequest(List<String> args) {
    try {
      JavaLibraryBuildRequest build = parse(args);
      AbstractJavaBuilder builder = build.getDependencyModule().reduceClasspath()
          ? new ReducedClasspathJavaLibraryBuilder()
          : new SimpleJavaLibraryBuilder();
      builder.run(build, System.err);
    } catch (JavacException | InvalidCommandLineException e) {
      System.err.println(CMDNAME + " threw exception: " + e.getMessage());
      return 1;
    } catch (Exception e) {
      e.printStackTrace();
      return 1;
    }
    return 0;
  }

  /**
   * Parses the list of arguments into a {@link JavaLibraryBuildRequest}. The returned
   * {@link JavaLibraryBuildRequest} object can be then used to configure the compilation itself.
   *
   * @throws IOException if the argument list contains a file (with the @ prefix) and reading that
   *         file failed
   * @throws InvalidCommandLineException on any command line error
   */
  @VisibleForTesting
  public static JavaLibraryBuildRequest parse(List<String> args) throws IOException,
      InvalidCommandLineException {
    ImmutableList<BlazeJavaCompilerPlugin> plugins =
        ImmutableList.<BlazeJavaCompilerPlugin>of(
            new ClassLoaderMaskingPlugin(),
            new ErrorPronePlugin());
    JavaLibraryBuildRequest build =
        new JavaLibraryBuildRequest(args, plugins, new DependencyModule.Builder());
    build.setJavacOpts(JavacOptions.normalizeOptions(build.getJavacOpts()));
    return build;
  }
}
