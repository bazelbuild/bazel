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

import com.android.dex.Dex;
import com.android.dex.DexFormat;
import com.android.dx.command.DxConsole;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
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

  /**
   * Commandline options.
   */
  public static class Options extends OptionsBase {
    @Option(
      name = "input",
      defaultValue = "null",
      category = "input",
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help = "Input file to read to aggregate."
    )
    public Path inputArchive;

    @Option(
      name = "output",
      defaultValue = "classes.dex.jar",
      category = "output",
      converter = PathConverter.class,
      abbrev = 'o',
      help = "Output archive to write."
    )
    public Path outputArchive;

    @Option(
      name = "multidex",
      defaultValue = "off",
      category = "multidex",
      converter = MultidexStrategyConverter.class,
      help = "Allow more than one .dex file in the output."
    )
    public MultidexStrategy multidexMode;

    @Option(
      name = "main-dex-list",
      defaultValue = "null",
      category = "multidex",
      converter = ExistingPathConverter.class,
      implicitRequirements = "--multidex=minimal",
      help = "List of classes to be placed into \"main\" classes.dex file."
    )
    public Path mainDexListFile;

    @Option(
      name = "minimal-main-dex",
      defaultValue = "false",
      category = "multidex",
      implicitRequirements = "--multidex=minimal",
      help =
          "If true, *only* classes listed in --main_dex_list file are placed into \"main\" "
              + "classes.dex file."
    )
    public boolean minimalMainDex;

    @Option(
      name = "verbose",
      defaultValue = "false",
      category = "misc",
      help = "If true, print information about the merged files and resulting files to stdout."
    )
    public boolean verbose;

    @Option(
      name = "max-bytes-wasted-per-file",
      defaultValue = "0",
      category = "misc",
      help =
          "Limit on conservatively allocated but unused bytes per dex file, which can enable "
              + "faster merging."
    )
    public int wasteThresholdPerDex;

    // Undocumented dx option for testing multidex logic
    @Option(
      name = "set-max-idx-number",
      defaultValue = "" + (DexFormat.MAX_MEMBER_IDX + 1),
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
      help = "Limit on fields and methods in a single dex file."
    )
    public int maxNumberOfIdxPerDex;
  }

  public static class MultidexStrategyConverter extends EnumConverter<MultidexStrategy> {
    public MultidexStrategyConverter() {
      super(MultidexStrategy.class, "multidex strategy");
    }
  }

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, Dexing.DexingOptions.class);
    optionsParser.parseAndExitUponError(args);

    buildMergedDexFiles(optionsParser.getOptions(Options.class));
  }

  @VisibleForTesting
  static void buildMergedDexFiles(Options options) throws IOException {
    ImmutableSet<String> classesInMainDex = options.mainDexListFile != null
        ? ImmutableSet.copyOf(Files.readAllLines(options.mainDexListFile, UTF_8))
        : null;
    PrintStream originalStdOut = System.out;
    try (ZipFile zip = new ZipFile(options.inputArchive.toFile());
        DexFileAggregator out = createDexFileAggregator(options)) {
      checkForUnprocessedClasses(zip);
      if (!options.verbose) {
        // com.android.dx.merge.DexMerger prints tons of debug information to System.out that we
        // silence here unless it was explicitly requested.
        System.setOut(DxConsole.noop);
      }

      if (classesInMainDex == null) {
        processDexFiles(zip, out, Predicates.<ZipEntry>alwaysTrue());
      } else {
        // Options parser should be making sure of this but let's be extra-safe as other modes
        // might result in classes from main dex list ending up in files other than classes.dex
        checkArgument(options.multidexMode == MultidexStrategy.MINIMAL, "Only minimal multidex "
            + "mode is supported with --main_dex_list, but mode is: %s", options.multidexMode);
        // To honor --main_dex_list make two passes:
        // 1. process only the classes listed in the given file
        // 2. process the remaining files
        Predicate<ZipEntry> classFileFilter = ZipEntryPredicates.classFileFilter(classesInMainDex);
        processDexFiles(zip, out, classFileFilter);
        // Fail if main_dex_list is too big, following dx's example
        checkState(out.getDexFilesWritten() == 0, "Too many classes listed in main dex list file "
            + "%s, main dex capacity exceeded", options.mainDexListFile);
        if (options.minimalMainDex) {
          out.flush(); // Start new .dex file if requested
        }
        processDexFiles(zip, out, Predicates.not(classFileFilter));
      }
    } finally {
      System.setOut(originalStdOut);
    }
    // Use input's timestamp for output file so the output file is stable.
    Files.setLastModifiedTime(options.outputArchive,
        Files.getLastModifiedTime(options.inputArchive));
  }

  private static void processDexFiles(
      ZipFile zip, DexFileAggregator out, Predicate<ZipEntry> extraFilter) throws IOException {
    @SuppressWarnings("unchecked") // Predicates.and uses varargs parameter with generics
    ArrayList<? extends ZipEntry> filesToProcess =
        Lists.newArrayList(
            Iterators.filter(
                Iterators.forEnumeration(zip.entries()),
                Predicates.and(
                    Predicates.not(ZipEntryPredicates.isDirectory()),
                    ZipEntryPredicates.suffixes(".dex"),
                    extraFilter)));
    Collections.sort(filesToProcess, ZipEntryComparator.LIKE_DX);
    for (ZipEntry entry : filesToProcess) {
      String filename = entry.getName();
      try (InputStream content = zip.getInputStream(entry)) {
        checkState(filename.endsWith(".dex"), "Shouldn't get here: %s", filename);
        // We don't want to use the Dex(InputStream) constructor because it closes the stream,
        // which will break the for loop, and it has its own bespoke way of reading the file into
        // a byte buffer before effectively calling Dex(byte[]) anyway.
        out.add(new Dex(ByteStreams.toByteArray(content)));
      }
    }
  }

  private static void checkForUnprocessedClasses(ZipFile zip) {
    Iterator<? extends ZipEntry> classes =
        Iterators.filter(
            Iterators.forEnumeration(zip.entries()),
            Predicates.and(
                Predicates.not(ZipEntryPredicates.isDirectory()),
                ZipEntryPredicates.suffixes(".class")));
    if (classes.hasNext()) {
      // Hitting this error indicates Jar files not covered by incremental dexing (b/34949364).
      // Bazel should prevent this error but if you do get this exception, you can use DexBuilder
      // to convert offending classes first. In Bazel that typically means using java_import or to
      // make sure Bazel rules use DexBuilder on implicit dependencies.
      throw new IllegalArgumentException(
          zip.getName()
              + " should only contain .dex files but found the following .class files: "
              + Iterators.toString(classes));
    }
  }

  private static DexFileAggregator createDexFileAggregator(Options options) throws IOException {
    return new DexFileAggregator(
        new DexFileArchive(
            new ZipOutputStream(
                new BufferedOutputStream(Files.newOutputStream(options.outputArchive)))),
        options.multidexMode,
        options.maxNumberOfIdxPerDex,
        options.wasteThresholdPerDex);
  }

  /**
   * Sorts java class names such that outer classes preceed their inner
   * classes and "package-info" preceeds all other classes in its package.
   *
   * @param a {@code non-null;} first class name
   * @param b {@code non-null;} second class name
   * @return {@code compareTo()}-style result
   */
  // Copied from com.android.dx.cf.direct.ClassPathOpener
  @VisibleForTesting
  static int compareClassNames(String a, String b) {
    // Ensure inner classes sort second
    a = a.replace('$', '0');
    b = b.replace('$', '0');

    /*
     * Assuming "package-info" only occurs at the end, ensures package-info
     * sorts first.
     */
    a = a.replace("package-info", "");
    b = b.replace("package-info", "");

    return a.compareTo(b);
  }

  /**
   * Comparator that orders {@link ZipEntry ZipEntries} {@link #LIKE_DX like Android's dx tool}.
   */
  private static enum ZipEntryComparator implements Comparator<ZipEntry> {
    /**
     * Comparator to order more or less order alphabetically by file name.  See
     * {@link DexFileMerger#compareClassNames} for the exact name comparison.
     */
    LIKE_DX;

    @Override
    // Copied from com.android.dx.cf.direct.ClassPathOpener
    public int compare (ZipEntry a, ZipEntry b) {
      return compareClassNames(a.getName(), b.getName());
    }
  }

  private DexFileMerger() {
  }
}
