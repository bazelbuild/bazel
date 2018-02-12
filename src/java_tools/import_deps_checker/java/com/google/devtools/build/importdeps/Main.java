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
import static com.google.common.io.Files.asCharSink;

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOException;
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
      defaultValue = "",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help = "Input jars with classes to check the completeness of their dependencies."
    )
    public List<Path> inputJars;

    @Option(
      name = "classpath_entry",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help =
          "Ordered classpath (Jar) to resolve symbols in the --input jars, like javac's -cp flag."
    )
    public List<Path> classpath;

    @Option(
      name = "bootclasspath_entry",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help =
          "Bootclasspath that was used to compile the --input Jar with, like javac's "
              + "-bootclasspath_entry flag (required)."
    )
    public List<Path> bootclasspath;

    @Option(
      name = "output",
      defaultValue = "",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      help = "Output path to save the result."
    )
    public Path output;
  }

  /**
   * A randomly picked large exit code to avoid collision with other common exit codes.
   */
  private static final int DEPS_ERROR_EXIT_CODE = 199;

  public static void main(String[] args) throws IOException {
    Options options = parseCommandLineOptions(args);

    if (!Files.exists(options.output)) {
      Files.createFile(options.output); // Make sure the output file always exists.
    }

    int exitCode = 0;
    try (ImportDepsChecker checker =
        new ImportDepsChecker(
            ImmutableList.copyOf(options.bootclasspath),
            ImmutableList.copyOf(options.classpath),
            ImmutableList.copyOf(options.inputJars))) {
      if (!checker.check()) {
        String result = checker.computeResultOutput();
        checkState(!result.isEmpty(), "The result should NOT be empty.");
        exitCode = DEPS_ERROR_EXIT_CODE;
        System.err.println(result);
        asCharSink(options.output.toFile(), StandardCharsets.UTF_8).write(result);
      }
    }
    System.exit(exitCode);
  }

  private static Options parseCommandLineOptions(String[] args) throws IOException {
    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.setAllowResidue(false);
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);

    checkArgument(!options.inputJars.isEmpty(), "--input is required");
    checkArgument(
        !options.classpath.isEmpty(), "--classpath_entry is required, at least the bootclasspath");
    checkArgument(!options.bootclasspath.isEmpty(), "--bootclasspath_entry is required");
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
}
