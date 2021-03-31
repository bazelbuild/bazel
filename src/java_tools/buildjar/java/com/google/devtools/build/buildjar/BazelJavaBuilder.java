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
import com.google.devtools.build.buildjar.javac.BlazeJavacResult.Status;
import com.google.devtools.build.buildjar.javac.FormattedDiagnostic;
import com.google.devtools.build.buildjar.javac.JavacOptions;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule;
import com.google.devtools.build.buildjar.javac.plugins.errorprone.ErrorPronePlugin;
import com.google.devtools.build.lib.worker.ProtoWorkerMessageProcessor;
import com.google.devtools.build.lib.worker.WorkRequestHandler;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;

/** The JavaBuilder main called by bazel. */
public class BazelJavaBuilder {

  private static final String CMDNAME = "BazelJavaBuilder";

  /** The main method of the BazelJavaBuilder. */
  public static void main(String[] args) {
    BazelJavaBuilder builder = new BazelJavaBuilder();
    if (args.length == 1 && args[0].equals("--persistent_worker")) {
      WorkRequestHandler workerHandler =
          new WorkRequestHandler(
              builder::parseAndBuild,
              System.err,
              new ProtoWorkerMessageProcessor(System.in, System.out),
              Duration.ofSeconds(10));
      try {
        workerHandler.processRequests();
      } catch (IOException e) {
        System.err.println(e.getMessage());
        System.exit(1);
      }
    } else {
      PrintWriter pw =
          new PrintWriter(new OutputStreamWriter(System.err, Charset.defaultCharset()));
      int returnCode;
      try {
        returnCode = builder.parseAndBuild(Arrays.asList(args), pw);
      } finally {
        pw.flush();
      }
      System.exit(returnCode);
    }
  }

  public int parseAndBuild(List<String> args, PrintWriter pw) {
    try {
      JavaLibraryBuildRequest build = parse(args);
      try (SimpleJavaLibraryBuilder builder =
          build.getDependencyModule().reduceClasspath()
              ? new ReducedClasspathJavaLibraryBuilder()
              : new SimpleJavaLibraryBuilder()) {

        return build(builder, build, pw);
      }
    } catch (InvalidCommandLineException e) {
      pw.println(CMDNAME + " threw exception: " + e.getMessage());
      return 1;
    } catch (Exception e) {
      e.printStackTrace();
      return 1;
    }
  }

  /**
   * Uses {@code builder} to build the target passed in {@code buildRequest}. All errors and
   * diagnostics should be written to {@code err}.
   *
   * @return An error code, 0 is success, any other value is an error.
   */
  protected int build(
      SimpleJavaLibraryBuilder builder, JavaLibraryBuildRequest buildRequest, Writer err)
      throws Exception {
    BlazeJavacResult result = builder.run(buildRequest);
    if (result.status() == Status.REQUIRES_FALLBACK) {
      return 0;
    }
    for (FormattedDiagnostic d : result.diagnostics()) {
      err.write(d.getFormatted() + "\n");
    }
    err.write(result.output());
    return result.isOk() ? 0 : 1;
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
  public JavaLibraryBuildRequest parse(List<String> args)
      throws IOException, InvalidCommandLineException {
    OptionsParser optionsParser =
        new OptionsParser(args, JavacOptions.createWithWarningsAsErrorsDefault(ImmutableList.of()));
    ImmutableList<BlazeJavaCompilerPlugin> plugins = ImmutableList.of(new ErrorPronePlugin());
    return new JavaLibraryBuildRequest(optionsParser, plugins, new DependencyModule.Builder());
  }
}
