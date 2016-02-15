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

package com.google.devtools.build.android.ideinfo;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.io.Files;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.JavaSourcePackage;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.PackageManifest;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * Parses the package string from each of the source .java files
 */
public class PackageParser {

  /** The options for a {@PackageParser} action. */
  public static final class PackageParserOptions extends OptionsBase {
    @Option(name = "sources_absolute_paths",
        defaultValue = "null",
        converter = PathListConverter.class,
        category = "input",
        help = "The absolute paths of the java source files. The expected format is a "
               + "colon-separated list.")
    public List<Path> sourcesAbsolutePaths;

    @Option(name = "sources_execution_paths",
        defaultValue = "null",
        converter = PathListConverter.class,
        category = "input",
        help = "The execution paths of the java source files. The expected format is a "
            + "colon-separated list.")
    public List<Path> sourcesExecutionPaths;

    @Option(name = "output_manifest",
        defaultValue = "null",
        converter = PathConverter.class,
        category = "output",
        help = "The path to the manifest file this parser writes to.")
    public Path outputManifest;
  }

  private static final Logger logger = Logger.getLogger(PackageParser.class.getName());

  private static final Pattern JAVA_PACKAGE_PATTERN =
      Pattern.compile("^\\s*package\\s+([\\w\\.]+);");

  public static void main(String[] args) throws Exception {
    PackageParserOptions options = parseArgs(args);
    Preconditions.checkNotNull(options.sourcesAbsolutePaths);
    Preconditions.checkNotNull(options.sourcesExecutionPaths);
    Preconditions.checkState(
        options.sourcesAbsolutePaths.size() == options.sourcesExecutionPaths.size());
    Preconditions.checkNotNull(options.outputManifest);

    try {
      PackageParser parser = new PackageParser(PackageParserIoProvider.INSTANCE);
      Map<Path, String> outputMap = parser.parsePackageStrings(options.sourcesAbsolutePaths,
          options.sourcesExecutionPaths);
      parser.writeManifest(outputMap, options.outputManifest);
    } catch (Throwable e) {
      logger.log(Level.SEVERE, "Error parsing package strings", e);
      System.exit(1);
    }
    System.exit(0);
  }

  @VisibleForTesting
  public static PackageParserOptions parseArgs(String[] args) {
    args = parseParamFileIfUsed(args);
    OptionsParser optionsParser = OptionsParser.newOptionsParser(PackageParserOptions.class);
    optionsParser.parseAndExitUponError(args);
    return optionsParser.getOptions(PackageParserOptions.class);
  }

  private static String[] parseParamFileIfUsed(@Nonnull String[] args) {
    if (args.length != 1 || !args[0].startsWith("@")) {
      return args;
    }
    File paramFile = new File(args[0].substring(1));
    try {
      return Files.readLines(paramFile, StandardCharsets.UTF_8).toArray(new String[0]);
    } catch (IOException e) {
      throw new RuntimeException("Error parsing param file: " + args[0], e);
    }
  }

  private final PackageParserIoProvider ioProvider;

  @VisibleForTesting
  public PackageParser(@Nonnull PackageParserIoProvider ioProvider) {
    this.ioProvider = ioProvider;
  }

  @VisibleForTesting
  public void writeManifest(@Nonnull Map<Path, String> sourceToPackageMap, Path outputFile)
      throws IOException {
    PackageManifest.Builder builder = PackageManifest.newBuilder();
    for (Entry<Path, String> entry : sourceToPackageMap.entrySet()) {
      builder.addSources(JavaSourcePackage.newBuilder()
          .setAbsolutePath(entry.getKey().toAbsolutePath().toString())
          .setPackageString(entry.getValue()));
    }

    try {
      ioProvider.writeProto(builder.build(), outputFile);
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Error writing package manifest", e);
      throw e;
    }
  }

  @Nonnull
  @VisibleForTesting
  public Map<Path, String> parsePackageStrings(@Nonnull List<Path> absolutePaths,
      @Nonnull List<Path> executionPaths) throws Exception {

    ListeningExecutorService executorService = MoreExecutors.listeningDecorator(
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()));

    Map<Path, ListenableFuture<String>> futures = Maps.newHashMap();
    for (int i = 0; i < absolutePaths.size(); i++) {
      final Path source = executionPaths.get(i);
      futures.put(absolutePaths.get(i), executorService.submit(new Callable<String>() {
        @Override
        public String call() throws Exception {
          return getDeclaredPackageOfJavaFile(source);
        }
      }));
    }
    Map<Path, String> map = Maps.newHashMap();
    for (Entry<Path, ListenableFuture<String>> entry : futures.entrySet()) {
      String value = entry.getValue().get();
      if (value != null) {
        map.put(entry.getKey(), value);
      }
    }
    return map;
  }

  @Nullable
  private String getDeclaredPackageOfJavaFile(@Nonnull Path source) {
    try (BufferedReader reader = ioProvider.getReader(source)) {
      return parseDeclaredPackage(reader);

    } catch (IOException e) {
      logger.log(Level.WARNING, "Error parsing package string from java source: " + source, e);
      return null;
    }
  }

  @VisibleForTesting
  @Nullable
  public static String parseDeclaredPackage(@Nonnull BufferedReader reader) throws IOException {
    String line;
    while ((line = reader.readLine()) != null) {
      Matcher packageMatch = JAVA_PACKAGE_PATTERN.matcher(line);
      if (packageMatch.find()) {
        return packageMatch.group(1);
      }
    }
    return null;
  }

}
