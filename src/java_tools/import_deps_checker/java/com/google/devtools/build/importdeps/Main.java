// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.importdeps;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.io.MoreFiles.asCharSink;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.view.proto.Deps.Dependencies;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionDef;
import org.kohsuke.args4j.OptionHandlerRegistry;
import org.kohsuke.args4j.spi.Parameters;
import org.kohsuke.args4j.spi.PathOptionHandler;
import org.kohsuke.args4j.spi.Setter;

/**
 * A checker that checks the completeness of the dependencies of an import target (java_import or
 * aar_import). If incomplete, it prints out the list of missing class names to the output file.
 */
public class Main {

  /** Command line options. */
  public static class Options {
    @Option(
        name = "--input",
        required = true,
        handler = ExistingPathOptionHandler.class,
        usage = "Input jars with classes to check the completeness of their dependencies.")
    public List<Path> inputJars = new ArrayList<>();

    @Option(
        name = "--directdep",
        handler = ExistingPathOptionHandler.class,
        usage =
            "Subset of Jars listed in --classpath_entry that --input Jars are allowed to depend "
                + "on directly.")
    public List<Path> directClasspath = new ArrayList<>();

    @Option(
        name = "--classpath_entry",
        handler = ExistingPathOptionHandler.class,
        usage =
            "Ordered classpath (Jar) to resolve symbols in the --input jars, like javac's -cp"
                + " flag.")
    public List<Path> fullClasspath = new ArrayList<>();

    @Option(
        name = "--bootclasspath_entry",
        required = true,
        handler = ExistingPathOptionHandler.class,
        usage =
            "Bootclasspath that was used to compile the --input Jar with, like javac's "
                + "-bootclasspath_entry flag (required).")
    public List<Path> bootclasspath = new ArrayList<>();

    @Option(
        name = "--output",
        handler = PathOptionHandler.class,
        usage = "Output path to save the result.")
    public Path output;

    @Option(
        name = "--jdeps_output",
        handler = PathOptionHandler.class,
        usage = "Output path to save the result.")
    public Path jdepsOutput;

    @Option(name = "--rule_label", usage = "The rule label of the current target under analysis.")
    public String ruleLabel = "";

    @Option(name = "--checking_mode", usage = "Controls the behavior of the checker.")
    public CheckingMode checkingMode = CheckingMode.WARNING;

    @Option(
        name = "--check_missing_members",
        handler = BooleanOptionHandler.class,
        usage = "Whether to check whether referenced fields and methods are defined.")
    public boolean checkMissingMembers = true;
  }

  /** A randomly picked large exit code to avoid collision with other common exit codes. */
  private static final int DEPS_ERROR_EXIT_CODE = 199;

  public static void main(String[] args) throws IOException {
    System.exit(checkDeps(args));
  }

  @VisibleForTesting
  static int checkDeps(String[] args) throws IOException {
    Options options = parseCommandLineOptions(args);

    if (options.output != null && !Files.exists(options.output)) {
      Files.createFile(options.output); // Make sure the output file always exists.
    }

    int exitCode = 0;
    try (ImportDepsChecker checker =
        new ImportDepsChecker(
            ImmutableSet.copyOf(options.bootclasspath),
            // Consider everything direct if no direct classpath is given
            options.directClasspath.isEmpty()
                ? ImmutableSet.copyOf(options.fullClasspath)
                : ImmutableSet.copyOf(options.directClasspath),
            ImmutableSet.copyOf(options.fullClasspath),
            ImmutableSet.copyOf(options.inputJars),
            options.checkMissingMembers)) {
      if (!checker.check() && options.checkingMode != CheckingMode.SILENCE) {
        String result = checker.computeResultOutput(options.ruleLabel);
        checkState(!result.isEmpty(), "The result should NOT be empty.");
        exitCode = options.checkingMode == CheckingMode.ERROR ? DEPS_ERROR_EXIT_CODE : 0;
        printErrorMessage(result, options);
        if (options.output != null) {
          asCharSink(options.output, StandardCharsets.UTF_8).write(result);
        }
      }
      if (options.jdepsOutput != null) {
        Dependencies dependencies = checker.emitJdepsProto(options.ruleLabel);
        try (OutputStream os =
            new BufferedOutputStream(Files.newOutputStream(options.jdepsOutput))) {
          dependencies.writeTo(os);
        }
      }
    }
    return exitCode;
  }

  private static void printErrorMessage(String detailedErrorMessage, Options options) {
    checkArgument(
        options.checkingMode == CheckingMode.ERROR || options.checkingMode == CheckingMode.WARNING);
    System.err.print(options.checkingMode == CheckingMode.ERROR ? "ERROR" : "WARNING");
    System.err.printf(
        ": The dependencies for the following %d jar(s) are not complete.\n",
        options.inputJars.size());
    int index = 1;
    for (Path jar : options.inputJars) {
      System.err.printf("    %3d.%s\n", index++, jar.toString());
    }
    System.err.println("The details are listed below:");
    System.err.print(detailedErrorMessage);
  }

  @VisibleForTesting
  static Options parseCommandLineOptions(String[] args) {
    Options options = new Options();
    CmdLineParser parser = new CmdLineParser(options);
    OptionHandlerRegistry.getRegistry().registerHandler(boolean.class, BooleanOptionHandler.class);
    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      throw new IllegalArgumentException(e);
    }

    checkArgument(!options.inputJars.isEmpty(), "--input is required");
    checkArgument(!options.bootclasspath.isEmpty(), "--bootclasspath_entry is required");
    // TODO(cushon): make --jdeps_output mandatory
    // checkArgument(
    //     options.jdepsOutput != null, "Invalid value of --jdeps_output: '%s'",
    //     options.jdepsOutput);

    return options;
  }

  /** The checking mode of the dependency checker. */
  public enum CheckingMode {
    /** Emit 'errors' on missing or incomplete dependencies. */
    ERROR,
    /** Emit 'warnings' on missing or incomplete dependencies. */
    WARNING,
    /**
     * Emit 'nothing' on missing or incomplete dependencies. This is mainly used to dump jdeps
     * protos.
     */
    SILENCE
  }

  /**
   * Custom option handler for boolean values, to support both "--foo=true" and "--foo" syntax. The
   * default args4j boolean handler only supports the latter.
   */
  public static class BooleanOptionHandler extends org.kohsuke.args4j.spi.BooleanOptionHandler {
    public BooleanOptionHandler(
        CmdLineParser parser, OptionDef option, Setter<? super Boolean> setter) {
      super(parser, option, setter);
    }

    @Override
    public int parseArguments(Parameters params) throws CmdLineException {
      if (params.size() > 0) {
        String value = params.getParameter(0);
        if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
          setter.addValue(Boolean.parseBoolean(value));
          return 1;
        }
      }
      return super.parseArguments(params);
    }
  }

  /** Custom option handler for a path that must exist. */
  public static class ExistingPathOptionHandler extends PathOptionHandler {

    public ExistingPathOptionHandler(
        CmdLineParser parser, OptionDef option, Setter<? super Path> setter) {
      super(parser, option, setter);
    }

    @Override
    protected Path parse(String argument) throws CmdLineException {
      Path path = FileSystems.getDefault().getPath(argument);
      if (!Files.exists(path)) {
        throw new CmdLineException(
            owner, String.format("Path %s for option %s does not exist.", argument, option));
      }
      return path;
    }
  }
}
