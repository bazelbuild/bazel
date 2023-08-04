// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import com.android.tools.r8.ByteDataView;
import com.android.tools.r8.CompilationFailedException;
import com.android.tools.r8.D8;
import com.android.tools.r8.D8Command;
import com.android.tools.r8.DexIndexedConsumer;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.origin.Origin;
import com.android.tools.r8.origin.PathOrigin;
import com.android.tools.r8.utils.ExceptionDiagnostic;
import com.android.tools.r8.utils.StringDiagnostic;
import com.google.common.collect.Maps;
import com.google.devtools.build.android.r8.OptionsConverters.ExistingPathConverter;
import com.google.devtools.build.android.r8.OptionsConverters.PathConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Tool used by Bazel as a replacement for Android's {@code dx} tool that assembles a single or, if
 * allowed and necessary, multiple {@code .dex} files from a given archive of {@code .dex} and
 * {@code .class} files. The tool merges the {@code .dex} files it encounters into a single file and
 * additionally encodes any {@code .class} files it encounters. If multidex is allowed then the tool
 * will generate multiple files subject to the {@code .dex} file format's limits on the number of
 * methods and fields.
 *
 * <p>D8 version of DexFileMerger.
 */
public class DexFileMerger {
  /** File name prefix of a {@code .dex} file automatically loaded in an archive. */
  private static final String DEX_PREFIX = "classes";

  private static final String DEFAULT_OUTPUT_ARCHIVE_FILENAME = "classes.dex.jar";

  private static final boolean PRINT_ARGS = false;

  private static final int NATIVE_MULTIDEX_API_LEVEL = 21;

  /** Strategies for outputting multiple {@code .dex} files supported by {@link DexFileMerger}. */
  public enum MultidexStrategy {
    /** Create exactly one .dex file. The operation will fail if .dex limits are exceeded. */
    OFF,
    /** Create exactly one &lt;prefixN&gt;.dex file with N taken from the (single) input archive. */
    GIVEN_SHARD,
    /**
     * Assemble .dex files similar to {@link com.android.dx.command.dexer.Main dx}, with all but one
     * file as large as possible.
     */
    MINIMAL,
    /**
     * Allow some leeway and sometimes use additional .dex files to speed up processing. This option
     * exists to give flexibility but it often (or always) may be identical to {@link #MINIMAL}.
     */
    BEST_EFFORT;

    public boolean isMultidexAllowed() {
      switch (this) {
        case OFF:
        case GIVEN_SHARD:
          return false;
        case MINIMAL:
        case BEST_EFFORT:
          return true;
      }
      throw new AssertionError("Unknown: " + this);
    }
  }

  /** Option converter for {@link MultidexStrategy}. */
  public static class MultidexStrategyConverter extends EnumConverter<MultidexStrategy> {
    public MultidexStrategyConverter() {
      super(MultidexStrategy.class, "multidex strategy");
    }
  }

  /** Commandline options. */
  public static class Options extends OptionsBase {
    @Option(
        name = "input",
        allowMultiple = true,
        defaultValue = "null",
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
        defaultValue = DEFAULT_OUTPUT_ARCHIVE_FILENAME,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = PathConverter.class,
        abbrev = 'o',
        help = "Output archive to write.")
    public Path outputArchive;

    @Option(
        name = "multidex",
        defaultValue = "off",
        category = "multidex",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = MultidexStrategyConverter.class,
        help = "Allow more than one .dex file in the output.")
    public MultidexStrategy multidexMode;

    @Option(
        name = "main-dex-list",
        defaultValue = "null",
        category = "multidex",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help = "List of classes to be placed into \"main\" classes.dex file.")
    public Path mainDexListFile;

    @Option(
        name = "verbose",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "If true, print information about the merged files and resulting files to stdout.")
    public boolean verbose;

    @Option(
        name = "dex_prefix",
        defaultValue = DEX_PREFIX,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Dex file output prefix.")
    public String dexPrefix;
  }

  private static Options parseArguments(String[] args) throws IOException {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);

    return optionsParser.getOptions(Options.class);
  }

  /**
   * Implements a DexIndexedConsumer writing into a ZipStream with support for custom dex file name
   * prefix, reindexing a single dex output file to a nonzero index and reporting if any data has
   * been written.
   */
  private static class ArchiveConsumer implements DexIndexedConsumer {
    private final Path path;
    private final String prefix;
    private final Integer singleFixedFileIndex;
    private final Origin origin;
    private ZipOutputStream stream = null;

    private int highestIndexWritten = -1;
    private final Map<Integer, Runnable> writers = new TreeMap<>();
    private boolean hasWrittenSomething = false;

    /** If singleFixedFileIndex is not null then we expect only one output dex file */
    private ArchiveConsumer(Path path, String prefix, Integer singleFixedFileIndex) {
      this.path = path;
      this.prefix = prefix;
      this.singleFixedFileIndex = singleFixedFileIndex;
      this.origin = new PathOrigin(path);
    }

    private boolean hasWrittenSomething() {
      return hasWrittenSomething;
    }

    private String getDexFileName(int fileIndex) {
      if (singleFixedFileIndex != null) {
        fileIndex = singleFixedFileIndex;
      }
      return prefix + (fileIndex == 0 ? "" : (fileIndex + 1)) + FileUtils.DEX_EXTENSION;
    }

    @Override
    public synchronized void accept(
        int fileIndex, ByteDataView data, Set<String> descriptors, DiagnosticsHandler handler) {
      if (singleFixedFileIndex != null && fileIndex != 0) {
        handler.error(new StringDiagnostic("Result does not fit into a single dex file."));
        return;
      }
      // Make a copy of the actual bytes as they will possibly be accessed later by the runner.
      final byte[] bytes = data.copyByteData();
      writers.put(fileIndex, () -> writeEntry(fileIndex, bytes, handler));

      while (writers.containsKey(highestIndexWritten + 1)) {
        ++highestIndexWritten;
        writers.get(highestIndexWritten).run();
        writers.remove(highestIndexWritten);
      }
    }

    /** Get or open the zip output stream. */
    private synchronized ZipOutputStream getStream(DiagnosticsHandler handler) {
      if (stream == null) {
        try {
          stream =
              new ZipOutputStream(
                  Files.newOutputStream(
                      path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING));
        } catch (IOException e) {
          handler.error(new ExceptionDiagnostic(e, origin));
        }
      }
      return stream;
    }

    private void writeEntry(int fileIndex, byte[] data, DiagnosticsHandler handler) {
      try {
        ZipUtils.writeToZipStream(
            getDexFileName(fileIndex),
            ByteDataView.of(data),
            ZipEntry.DEFLATED,
            getStream(handler));
        hasWrittenSomething = true;
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, origin));
      }
    }

    @Override
    public void finished(DiagnosticsHandler handler) {
      if (!writers.isEmpty()) {
        handler.error(
            new StringDiagnostic(
                "Failed to write zip, for a multidex output some of the classes.dex files were"
                    + " not produced."));
      }
      try {
        if (stream != null) {
          stream.close();
          stream = null;
        }
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, origin));
      }
    }
  }

  private static int parseFileIndexFromShardFilename(String inputArchive) {
    Pattern namingPattern = Pattern.compile("([0-9]+)\\..*");
    String name = new File(inputArchive).getName();
    Matcher matcher = namingPattern.matcher(name);
    if (!matcher.matches()) {
      throw new IllegalStateException(
          String.format(
              "Expect input named <N>.xxx.zip for --multidex=given_shard but got %s.", name));
    }
    int shard = Integer.parseInt(matcher.group(1));
    if (shard <= 0) {
      throw new IllegalStateException(
          String.format("Expect positive N in input named <N>.xxx.zip but got %d.", shard));
    }
    return shard;
  }

  private static int getInputNumber(
      Origin origin, Map<String, Integer> inputOrdering, DiagnosticsHandler handler) {
    Integer number = inputOrdering.get(origin.parent().part());
    if (number == null) {
      String message = "Class parent path not found among input paths: " + origin;
      handler.error(new StringDiagnostic(message, origin));
      throw new IllegalStateException(message);
    }
    return number;
  }

  public static void run(String[] args) throws CompilationFailedException, IOException {
    Options options = parseArguments(args);

    if (options.inputArchives.isEmpty()) {
      throw new IllegalStateException("Need at least one --input");
    }

    if (options.mainDexListFile != null && options.inputArchives.size() != 1) {
      throw new IllegalStateException(
          "--main-dex-list only supported with exactly one --input, use DexFileSplitter for more");
    }

    if (!options.multidexMode.isMultidexAllowed()) {
      if (options.mainDexListFile != null) {
        throw new IllegalStateException(
            "--main-dex-list is only supported with multidex enabled, but mode is: "
                + options.multidexMode);
      }
    }

    D8Command.Builder builder = D8Command.builder();

    // If multidex is enabled but no main-dex list given then the build must be targeting
    // devices with native multidex support.
    if (options.multidexMode.isMultidexAllowed() && options.mainDexListFile == null) {
      builder.setMinApiLevel(NATIVE_MULTIDEX_API_LEVEL);
    }

    // The merge step assumes that no further desugaring is needed.
    builder = builder.setDisableDesugaring(true);

    // D8 does not allow duplicate classes. Resolve conflicts based on input order.
    Map<String, Integer> inputOrdering =
        Maps.newHashMapWithExpectedSize(options.inputArchives.size());
    int sequenceNumber = 0;
    for (Path s : options.inputArchives) {
      builder = builder.addProgramFiles(s);
      inputOrdering.put(s.toString(), sequenceNumber++);
    }
    builder =
        builder.setClassConflictResolver(
            (reference, origins, handler) -> {
              Iterator<Origin> it = origins.iterator();
              Origin first = it.next();
              int firstNumber = getInputNumber(first, inputOrdering, handler);
              while (it.hasNext()) {
                Origin next = it.next();
                int nextNumber = getInputNumber(next, inputOrdering, handler);
                if (firstNumber > nextNumber) {
                  first = next;
                  firstNumber = nextNumber;
                }
              }
              return first;
            });

    // Determine enabling multidexing and file indexing.
    Integer singleFixedFileIndex = null;
    switch (options.multidexMode) {
      case OFF:
        singleFixedFileIndex = 0;
        break;
      case GIVEN_SHARD:
        if (options.inputArchives.size() != 1) {
          throw new IllegalStateException("'--multidex=given_shard' requires exactly one --input.");
        }
        singleFixedFileIndex =
            parseFileIndexFromShardFilename(options.inputArchives.get(0).toString()) - 1;
        break;
      case MINIMAL:
      case BEST_EFFORT:
        // Nothing to do.
        break;
    }

    if (builder.getMinApiLevel() < NATIVE_MULTIDEX_API_LEVEL && options.mainDexListFile != null) {
      builder = builder.addMainDexListFiles(options.mainDexListFile);
    }

    ArchiveConsumer consumer =
        new ArchiveConsumer(options.outputArchive, options.dexPrefix, singleFixedFileIndex);
    builder = builder.setProgramConsumer(consumer);

    D8.run(builder.build());

    // If input was empty we still need to write out an empty zip.
    if (!consumer.hasWrittenSomething()) {
      File f = options.outputArchive.toFile();
      try (ZipOutputStream out = new ZipOutputStream(new FileOutputStream(f))) {}
    }
  }

  public static void main(String[] args) {
    try {
      if (PRINT_ARGS) {
        printArgs(args);
      }
      run(args);
    } catch (CompilationFailedException | IOException e) {
      System.err.println("Merge failed: " + e.getMessage());
      throw new RuntimeException("Merge failed: " + e.getMessage(), e);
    }
  }

  private static void printArgs(String[] args) {
    System.err.print("r8.DexFileMerger");
    for (String s : args) {
      System.err.printf(" %s", s);
    }
    System.err.println();
  }

  private DexFileMerger() {}
}
