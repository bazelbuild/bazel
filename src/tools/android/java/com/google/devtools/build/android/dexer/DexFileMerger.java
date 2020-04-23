// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.android.dex.Dex;
import com.android.dex.DexFormat;
import com.android.dx.command.dexer.DxContext;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Tool used by Bazel as a replacement for Android's {@code dx} tool that assembles a single or, if
 * allowed and necessary, multiple {@code .dex} files from a given archive of {@code .dex} and
 * {@code .class} files.  The tool merges the {@code .dex} files it encounters into a single file
 * and additionally encodes any {@code .class} files it encounters.  If multidex is allowed then the
 * tool will generate multiple files subject to the {@code .dex} file format's limits on the number
 * of methods and fields.
 */
class DexFileMerger {

  /** File name prefix of a {@code .dex} file automatically loaded in an archive. */
  private static final String DEX_PREFIX = "classes";

  /**
   * Commandline options.
   */
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
        help =
            "Input archives with .dex files to merge.  Inputs are processed in given order, so"
                + " classes from later inputs will be added after earlier inputs.  Duplicate"
                + " classes are dropped.")
    public List<Path> inputArchives;

    @Option(
      name = "output",
      defaultValue = "classes.dex.jar",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      abbrev = 'o',
      help = "Output archive to write."
    )
    public Path outputArchive;

    @Option(
      name = "multidex",
      defaultValue = "off",
      category = "multidex",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = MultidexStrategyConverter.class,
      help = "Allow more than one .dex file in the output."
    )
    public MultidexStrategy multidexMode;

    @Option(
      name = "main-dex-list",
      defaultValue = "null",
      category = "multidex",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help = "List of classes to be placed into \"main\" classes.dex file."
    )
    public Path mainDexListFile;

    @Option(
      name = "minimal-main-dex",
      defaultValue = "false",
      category = "multidex",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "If true, *only* classes listed in --main_dex_list file are placed into \"main\" "
              + "classes.dex file."
    )
    public boolean minimalMainDex;

    @Option(
      name = "verbose",
      defaultValue = "false",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "If true, print information about the merged files and resulting files to stdout."
    )
    public boolean verbose;

    @Option(
      name = "max-bytes-wasted-per-file",
      defaultValue = "0",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Limit on conservatively allocated but unused bytes per dex file, which can enable "
              + "faster merging."
    )
    public int wasteThresholdPerDex;

    // Undocumented dx option for testing multidex logic
    @Option(
      name = "set-max-idx-number",
      defaultValue = "" + DexFormat.MAX_MEMBER_IDX,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Limit on fields and methods in a single dex file."
    )
    public int maxNumberOfIdxPerDex;

    @Option(
      name = "forceJumbo",
      defaultValue = "false", // dx's default
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Typically not needed flag intended to imitate dx's --forceJumbo."
    )
    public boolean forceJumbo;

    @Option(
      name = "dex_prefix",
      defaultValue = DEX_PREFIX, // dx's default
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Dex file output prefix."
    )
    public String dexPrefix;
  }

  public static class MultidexStrategyConverter extends EnumConverter<MultidexStrategy> {
    public MultidexStrategyConverter() {
      super(MultidexStrategy.class, "multidex strategy");
    }
  }

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);

    buildMergedDexFiles(optionsParser.getOptions(Options.class));
  }

  @VisibleForTesting
  static void buildMergedDexFiles(Options options) throws IOException {
    ListeningExecutorService executor;
    checkArgument(!options.inputArchives.isEmpty(), "Need at least one --input");
    checkArgument(
        options.mainDexListFile == null || options.inputArchives.size() == 1,
        "--main-dex-list only supported with exactly one --input, use DexFileSplitter for more");
    if (options.multidexMode.isMultidexAllowed()) {
      executor = createThreadPool();
    } else {
      checkArgument(
          options.mainDexListFile == null,
          "--main-dex-list is only supported with multidex enabled, but mode is: %s",
          options.multidexMode);
      checkArgument(
          !options.minimalMainDex,
          "--minimal-main-dex is only supported with multidex enabled, but mode is: %s",
          options.multidexMode);
      // We'll only ever merge and write one dex file, so multi-threading is pointless.
      executor = MoreExecutors.newDirectExecutorService();
    }

    ImmutableSet<String> classesInMainDex = options.mainDexListFile != null
        ? ImmutableSet.copyOf(Files.readAllLines(options.mainDexListFile, UTF_8))
        : null;
    PrintStream originalStdOut = System.out;
    try (DexFileAggregator out = createDexFileAggregator(options, executor)) {
      if (!options.verbose) {
        // com.android.dx.merge.DexMerger prints status information to System.out that we silence
        // here unless it was explicitly requested.  (It also prints debug info to DxContext.out,
        // which we populate accordingly below.)
        System.setOut(Dexing.nullout);
      }

      LinkedHashSet<String> seen = new LinkedHashSet<>();
      for (Path inputArchive : options.inputArchives) {
        // Simply merge files from inputs in order.  Doing that with a main dex list doesn't work,
        // but we rule out more than one input with a main dex list above.
        try (ZipFile zip = new ZipFile(inputArchive.toFile())) {
          ArrayList<ZipEntry> dexFiles = filesToProcess(zip);
          if (classesInMainDex == null) {
            processDexFiles(zip, dexFiles, seen, out);
          } else {
            // To honor --main_dex_list make two passes:
            // 1. process only the classes listed in the given file
            // 2. process the remaining files
            Predicate<ZipEntry> mainDexFilter =
                ZipEntryPredicates.classFileFilter(classesInMainDex);
            processDexFiles(zip, Iterables.filter(dexFiles, mainDexFilter), seen, out);
            // Fail if main_dex_list is too big, following dx's example
            checkState(out.getDexFilesWritten() == 0, "Too many classes listed in main dex list "
                + "file %s, main dex capacity exceeded", options.mainDexListFile);
            if (options.minimalMainDex) {
              out.flush(); // Start new .dex file if requested
            }
            processDexFiles(
                zip, Iterables.filter(dexFiles, Predicates.not(mainDexFilter)), seen, out);
          }
        }
      }
    } finally {
      // Kill threads in the pool so we don't hang
      MoreExecutors.shutdownAndAwaitTermination(executor, 1, SECONDS);
      System.setOut(originalStdOut);
    }
  }

  /**
   * Returns all .dex and .class files in the given zip.  .class files are unexpected but we'll
   * deal with them later.
   */
  private static ArrayList<ZipEntry> filesToProcess(ZipFile zip) {
    ArrayList<ZipEntry> result = Lists.newArrayList(
        Iterators.filter(
            Iterators.forEnumeration(zip.entries()),
            Predicates.and(
                Predicates.not(ZipEntryPredicates.isDirectory()),
                ZipEntryPredicates.suffixes(".dex", ".class"))));
    Collections.sort(result, ZipEntryComparator.LIKE_DX);
    return result;
  }

  private static void processDexFiles(
      ZipFile zip,
      Iterable<ZipEntry> filesToProcess,
      LinkedHashSet<String> seen,
      DexFileAggregator out)
      throws IOException {
    for (ZipEntry entry : filesToProcess) {
      String filename = entry.getName();
      checkState(filename.endsWith(".dex"), "Input shouldn't contain .class files: %s", filename);
      if (!seen.add(filename)) {
        continue;  // pick first occurrence of each file to match how JVM treats dupes on classpath
      }
      try (InputStream content = zip.getInputStream(entry)) {
        // We don't want to use the Dex(InputStream) constructor because it closes the stream,
        // which will break the for loop, and it has its own bespoke way of reading the file into
        // a byte buffer before effectively calling Dex(byte[]) anyway.
        out.add(new Dex(ByteStreams.toByteArray(content)));
      }
    }
  }

  private static DexFileAggregator createDexFileAggregator(
      Options options, ListeningExecutorService executor) throws IOException {
    String filePrefix = options.dexPrefix;
    if (options.multidexMode == MultidexStrategy.GIVEN_SHARD) {
      checkArgument(options.inputArchives.size() == 1,
          "--multidex=given_shard requires exactly one --input");
      Pattern namingPattern = Pattern.compile("([0-9]+)\\..*");
      Matcher matcher = namingPattern.matcher(options.inputArchives.get(0).toFile().getName());
      checkArgument(matcher.matches(),
          "expect input named <N>.xxx.zip for --multidex=given_shard but got %s",
          options.inputArchives.get(0).toFile().getName());
      int shard = Integer.parseInt(matcher.group(1));
      checkArgument(shard > 0, "expect positive N in input named <N>.xxx.zip but got %s", shard);
      if (shard > 1) {  // first shard conventionally isn't numbered
        filePrefix += shard;
      }
    }
    return new DexFileAggregator(
        new DxContext(options.verbose ? System.out : ByteStreams.nullOutputStream(), System.err),
        new DexFileArchive(
            new ZipOutputStream(
                new BufferedOutputStream(Files.newOutputStream(options.outputArchive)))),
        executor,
        options.multidexMode,
        options.forceJumbo,
        options.maxNumberOfIdxPerDex,
        options.wasteThresholdPerDex,
        filePrefix);
  }

  /**
   * Creates an unbounded thread pool executor, which is appropriate here since the number of tasks
   * we will add to the thread pool is at most dozens and some of them perform I/O (ie, may block).
   */
  private static ListeningExecutorService createThreadPool() {
    return MoreExecutors.listeningDecorator(Executors.newCachedThreadPool());
  }

  private DexFileMerger() {
  }
}
