// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.android.dex.Dex;
import com.android.dex.DexFormat;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.TreeMultimap;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closer;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;

/**
 * Shuffles .class.dex files from input archives into 1 or more archives each to be merged into a
 * single final .dex file by {@link DexFileMerger}, respecting main dex list and other constraints
 * similar to how dx would process these files if they were in a single input archive.
 */
class DexFileSplitter implements Closeable {

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
        help = "Input dex archive.")
    public List<Path> inputArchives;

    @Option(
      name = "output",
      defaultValue = ".",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      abbrev = 'o',
      help = "Directory to write dex archives to merge."
    )
    public Path outputDirectory;

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
      name = "inclusion_filter_jar",
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help = "If given, only classes in the given Jar are included in outputs."
    )
    public Path inclusionFilterJar;
  }

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class)
            .allowResidue(false)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);

    splitIntoShards(optionsParser.getOptions(Options.class));
  }

  @VisibleForTesting
  static void splitIntoShards(Options options) throws IOException {
    checkArgument(
        !options.minimalMainDex || options.mainDexListFile != null,
        "--minimal-main-dex not allowed without --main-dex-list");

    if (!Files.exists(options.outputDirectory)) {
      Files.createDirectories(options.outputDirectory);
    }

    ImmutableSet<String> classesInMainDex =
        options.mainDexListFile != null
            ? ImmutableSet.copyOf(Files.readAllLines(options.mainDexListFile, UTF_8))
            : null;
    ImmutableSet<String> expected =
        options.inclusionFilterJar != null ? expectedEntries(options.inclusionFilterJar) : null;
    try (Closer closer = Closer.create();
        DexFileSplitter out =
            new DexFileSplitter(options.outputDirectory, options.maxNumberOfIdxPerDex)) {

      // 1. Scan inputs in order and keep first occurrence of each class, keeping all zips open.
      // We don't process anything yet so we can shard in sorted order, which is what dx would do
      // if presented with a single jar containing all the given inputs.
      // TODO(kmb): Abandon alphabetic sorting to process each input fully before moving on (still
      // requires scanning inputs twice for main dex list).

      Predicate<ZipEntry> inclusionFilter = ZipEntryPredicates.suffixes(".dex", ".class");
      if (expected != null) {
        inclusionFilter = inclusionFilter.and(e -> expected.contains(e.getName()));
      }

      // Maps a dex file name to the zip file containing that dex file.
      TreeMap<String, ZipFile> dexFilesAndContainingZip =
          new TreeMap<>(ZipEntryComparator::compareClassNames);
      // Maps a class to its synthetic classes, if any.
      TreeMultimap<String, String> contextClassesToSyntheticClasses = TreeMultimap.create();

      for (Path inputArchive : options.inputArchives) {
        ZipFile zip = closer.register(new ZipFile(inputArchive.toFile()));

        // synthetic-contexts.map is generated by CompatDexBuilder.
        ZipEntry syntheticContextsZipEntry = zip.getEntry("META-INF/synthetic-contexts.map");
        if (syntheticContextsZipEntry != null) {
          parseSyntheticContextsMap(
              zip.getInputStream(syntheticContextsZipEntry), contextClassesToSyntheticClasses);
        }

        zip.stream()
            .filter(inclusionFilter)
            .forEach(e -> dexFilesAndContainingZip.putIfAbsent(e.getName(), zip));
      }

      // 2. Process each class in desired order, rolling from shard to shard as needed.
      if (classesInMainDex == null || classesInMainDex.isEmpty()) {
        out.processDexes(
            dexFilesAndContainingZip, contextClassesToSyntheticClasses, Predicates.alwaysTrue());
      } else {
        checkArgument(classesInMainDex.stream().noneMatch(s -> s.startsWith("j$/")),
            "%s lists classes in package 'j$', which can't be included in classes.dex and can "
                + "cause runtime errors. Please avoid needing these classes in the main dex file.",
            options.mainDexListFile);
        // To honor --main_dex_list make two passes:
        // 1. process only the classes listed in the given file
        // 2. process the remaining files
        Predicate<String> mainDexFilter = ZipEntryPredicates.classFileNameFilter(classesInMainDex);
        out.processDexes(dexFilesAndContainingZip, contextClassesToSyntheticClasses, mainDexFilter);
        // Fail if main_dex_list is too big, following dx's example
        checkState(out.shardsWritten() == 0, "Too many classes listed in main dex list file "
            + "%s, main dex capacity exceeded", options.mainDexListFile);
        if (options.minimalMainDex) {
          out.nextShard(); // Start new .dex file if requested
        }
        out.processDexes(
            dexFilesAndContainingZip, contextClassesToSyntheticClasses, mainDexFilter.negate());
      }
    }
  }

  private static ImmutableSet<String> expectedEntries(Path filterJar) throws IOException {
    try (ZipFile zip = new ZipFile(filterJar.toFile())) {
      return zip.stream()
          .filter(ZipEntryPredicates.suffixes(".class"))
          .map(e -> e.getName() + ".dex")
          .collect(ImmutableSet.toImmutableSet());
    }
  }

  private static void parseSyntheticContextsMap(
      InputStream inputStream, TreeMultimap<String, String> syntheticClassContexts) {
    Scanner scanner = new Scanner(inputStream, UTF_8);
    scanner.useDelimiter("[;\n]");
    while (scanner.hasNext()) {
      String syntheticClass = scanner.next();
      String context = scanner.next();
      // DexFileSplitter mostly expects filenames which all end in .class.dex, while the synthetic
      // context map has class names, so add the extension here to make this easier to work with in
      // the rest of the code.
      syntheticClassContexts.put(
          context + CLASS_DEX_EXTENSION, syntheticClass + CLASS_DEX_EXTENSION);
    }
  }

  private static final String CLASS_DEX_EXTENSION = ".class.dex";

  private final int maxNumberOfIdxPerDex;
  private final Path outputDirectory;
  /** Collect written zip files so we can conveniently wait for all of them to close when done. */
  private final Closer closer = Closer.create();

  private int curShard = 0;
  /** Currently written file. */
  private AsyncZipOut curOut;
  private DexLimitTracker tracker;
  private Boolean inCoreLib;

  @SuppressWarnings("LenientFormatStringValidation")
  private DexFileSplitter(Path outputDirectory, int maxNumberOfIdxPerDex) throws IOException {
    // Expected 0 args, but got 1.
    checkArgument(!Files.isRegularFile(outputDirectory), "Must be a directory: ", outputDirectory);
    this.maxNumberOfIdxPerDex = maxNumberOfIdxPerDex;
    this.outputDirectory = outputDirectory;
    startShard();
  }

  private void nextShard() throws IOException {
    // Eagerly tell the last shard that it's done so it can finish writing the zip file and release
    // resources as soon as possible, without blocking the start of the next shard.
    curOut.finishAsync();  // will NPE if called after close()
    ++curShard;
    startShard();
  }

  private void startShard() throws IOException {
    tracker = new DexLimitTracker(maxNumberOfIdxPerDex);
    curOut =
        closer.register(
            new AsyncZipOut(
                outputDirectory.resolve((curShard + 1) + ".shard.zip"),
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE));
  }

  private int shardsWritten() {
    return curShard;
  }

  @Override
  public void close() throws IOException {
    if (curOut != null) {
      curOut.finishAsync();
      curOut = null;
      ++curShard;
    }
    // Wait for all shards to finish writing.  We told them to finish already but need to wait for
    // any pending writes so we're sure all output was successfully written.
    closer.close();
  }

  private void processDexes(
      Map<String, ZipFile> dexFilesAndContainingZip,
      TreeMultimap<String, String> contextClassesToSyntheticClasses,
      Predicate<String> filter)
      throws IOException {

    Set<String> syntheticClasses = new HashSet<>(contextClassesToSyntheticClasses.values());
    for (Map.Entry<String, ZipFile> entry : dexFilesAndContainingZip.entrySet()) {
      String filename = entry.getKey();
      if (filter.test(filename)) {
        // Synthetic classes will be gathered with their context classes and added to the dex file
        // all together as a unit, so skip them here.
        if (!syntheticClasses.contains(filename)) {
          ZipFile zipFile = entry.getValue();
          processDex(zipFile, filename, contextClassesToSyntheticClasses.get(filename));
        }
      }
    }
  }

  private void processDex(ZipFile zip, String filename, Set<String> syntheticClasses)
      throws IOException {

    // Synthetic classes base their names on their context classes, so this check only needs to be
    // done for the context class.
    if (inCoreLib == null) {
      inCoreLib = filename.startsWith("j$/");
    } else if (inCoreLib != filename.startsWith("j$/")) {
      // Put j$.xxx classes in separate file.  This shouldn't normally happen (b/134705306).
      nextShard();
      inCoreLib = !inCoreLib;
    }
    if (inCoreLib) {
      System.err.printf(
          "WARNING: Unexpected file %s found. Please ensure this only happens in test APKs.%n",
          filename);
    }

    List<ZipEntryDexAndContent> zipEntryDexAndContents = new ArrayList<>();
    ZipEntryDexAndContent contextZdc = processDex(zip, filename);
    checkNotNull(contextZdc, "Context class %s expected to be in %s", filename, zip.getName());
    zipEntryDexAndContents.add(contextZdc);

    for (String syntheticClass : syntheticClasses) {
      ZipEntryDexAndContent syntheticClassZdc = processDex(zip, syntheticClass);
      // Some synthetic classes are contained within the same dex as their enclosing class,
      // so they won't be standalone dexes in the zip file, and some synthetic classes are present
      // in synthetic-contexts.map but aren't standalone dexes in the zip nor are they in the
      // dex with their enclosing class, so just skip these.
      if (syntheticClassZdc != null) {
        zipEntryDexAndContents.add(syntheticClassZdc);
      }
    }

    if (tracker.outsideLimits()) {
      nextShard();
      for (ZipEntryDexAndContent zdc : zipEntryDexAndContents) {
        tracker.track(zdc.dex);
      }
      checkState(
          !tracker.outsideLimits(),
          "Impossible to fit %s and all of its synthetic classes (count: %s) in a single shard",
          filename,
          syntheticClasses.size());
    }

    for (ZipEntryDexAndContent zdc : zipEntryDexAndContents) {
      curOut.writeAsync(zdc.zipEntry, zdc.content);
    }
  }

  @Nullable
  private ZipEntryDexAndContent processDex(ZipFile zip, String filename) throws IOException {
    ZipEntry entry = zip.getEntry(filename);
    if (entry == null) {
      return null;
    }

    checkState(
        filename.endsWith(CLASS_DEX_EXTENSION),
        "%s isn't a dex archive: %s",
        zip.getName(),
        filename);
    checkState(entry.getMethod() == ZipEntry.STORED, "Expect to process STORED: %s", filename);

    try (InputStream entryStream = zip.getInputStream(entry)) {
      // We don't want to use the Dex(InputStream) constructor because it closes the stream,
      // which will break the for loop, and it has its own bespoke way of reading the file into
      // a byte buffer before effectively calling Dex(byte[]) anyway.
      // TODO(kmb) since entry is stored, mmap content and give to Dex(ByteBuffer) and output zip
      byte[] content = new byte[(int) entry.getSize()];
      ByteStreams.readFully(entryStream, content); // throws if file is smaller than expected
      checkState(
          entryStream.read() == -1,
          "Too many bytes in jar entry %s, expected %s",
          entry,
          entry.getSize());

      Dex dexFile = new Dex(content);
      tracker.track(dexFile);
      return new ZipEntryDexAndContent(entry, content, dexFile);
    }
  }

  private static final class ZipEntryDexAndContent {
    final ZipEntry zipEntry;
    final byte[] content;
    final Dex dex;

    ZipEntryDexAndContent(ZipEntry zipEntry, byte[] content, Dex dex) {
      this.zipEntry = zipEntry;
      this.content = content;
      this.dex = dex;
    }
  }
}
