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
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.List;

/**
 * A checker that checks the completeness of the dependencies of an import target (java_import or
 * aar_import). If incomplete, it prints out the list of missing class names to the output file.
 */
public class Main {

  /** Command line options. */
  public static class Options extends OptionsBase {
    @Option(
        name = "input",
        allowMultiple = true,
        defaultValue = "null",
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        abbrev = 'i',
        help = "Input jars with classes to check the completeness of their dependencies.")
    public List<Path> inputJars;

    @Option(
        name = "directdep",
        allowMultiple = true,
        defaultValue = "null",
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help =
            "Subset of Jars listed in --classpath_entry that --input Jars are allowed to depend "
                + "on directly.")
    public List<Path> directClasspath;

    @Option(
        name = "classpath_entry",
        allowMultiple = true,
        defaultValue = "null",
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help =
            "Ordered classpath (Jar) to resolve symbols in the --input jars, like javac's -cp"
                + " flag.")
    public List<Path> fullClasspath;

    @Option(
        name = "bootclasspath_entry",
        allowMultiple = true,
        defaultValue = "null",
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help =
            "Bootclasspath that was used to compile the --input Jar with, like javac's "
                + "-bootclasspath_entry flag (required).")
    public List<Path> bootclasspath;

    @Option(
        name = "output",
        defaultValue = "null",
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = PathConverter.class,
        help = "Output path to save the result.")
    public Path output;

    @Option(
        name = "jdeps_output",
        defaultValue = "null",
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = PathConverter.class,
        help = "Output path to save the result.")
    public Path jdepsOutput;

    @Option(
        name = "rule_label",
        defaultValue = "",
        category = "output",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "The rule label of the current target under analysis.")
    public String ruleLabel;

    @Option(
        name = "checking_mode",
        defaultValue = "WARNING",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = CheckingModeConverter.class,
        help = "Controls the behavior of the checker.")
    public CheckingMode checkingMode;

    @Option(
        name = "check_missing_members",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Whether to check whether referenced fields and methods are defined.")
    public boolean checkMissingMembers;
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
  static Options parseCommandLineOptions(String[] args) throws IOException {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class)
            .allowResidue(false)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);

    checkArgument(!options.inputJars.isEmpty(), "--input is required");
    checkArgument(!options.bootclasspath.isEmpty(), "--bootclasspath_entry is required");
    // TODO(cushon): make --jdeps_output mandatory
    // checkArgument(
    //     options.jdepsOutput != null, "Invalid value of --jdeps_output: '%s'",
    //     options.jdepsOutput);

    return options;
  }

  /** Validating converter for Paths. A Path is considered valid if it resolves to a file. */
  public static class PathConverter implements Converter<Path> {

    private final boolean mustExist;

    public PathConverter() {
      this.mustExist = false;
    }

    protected PathConverter(boolean mustExist) {
      this.mustExist = mustExist;
    }

    @Override
    public Path convert(String input) throws OptionsParsingException {
      try {
        Path path = FileSystems.getDefault().getPath(input);
        if (mustExist && !Files.exists(path)) {
          throw new OptionsParsingException(
              String.format("%s is not a valid path: it does not exist.", input));
        }
        return path;
      } catch (InvalidPathException e) {
        throw new OptionsParsingException(
            String.format("%s is not a valid path: %s.", input, e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a valid filesystem path";
    }
  }

  /**
   * Validating converter for Paths. A Path is considered valid if it resolves to a file and exists.
   */
  public static class ExistingPathConverter extends PathConverter {
    public ExistingPathConverter() {
      super(true);
    }
  }

  /** Converter for {@link CheckingMode} */
  public static class CheckingModeConverter extends EnumConverter<CheckingMode> {
    public CheckingModeConverter() {
      super(CheckingMode.class, "The checking mode for the dependency checker.");
    }
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
}
