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

import static com.google.common.base.Verify.verify;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toList;

import com.android.tools.r8.ByteDataView;
import com.android.tools.r8.CompatDxSupport;
import com.android.tools.r8.CompilationFailedException;
import com.android.tools.r8.CompilationMode;
import com.android.tools.r8.D8;
import com.android.tools.r8.D8Command;
import com.android.tools.r8.DexIndexedConsumer;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.ProgramConsumer;
import com.android.tools.r8.Version;
import com.android.tools.r8.errors.CompilationError;
import com.android.tools.r8.errors.Unimplemented;
import com.android.tools.r8.origin.Origin;
import com.android.tools.r8.origin.PathOrigin;
import com.android.tools.r8.utils.ArchiveResourceProvider;
import com.android.tools.r8.utils.ExceptionDiagnostic;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.r8.CompatDx.DxCompatOptions.DxUsageMessage;
import com.google.devtools.build.android.r8.CompatDx.DxCompatOptions.PositionInfo;
import com.google.devtools.common.options.Converters.StringConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Dx compatibility interface for d8.
 *
 * <p>This should become a mostly drop-in replacement for uses of the DX dexer (eg, dx --dex ...).
 */
public class CompatDx {

  private static final String USAGE_HEADER = "Usage: compatdx [options] <input files>";

  /** Commandline options. */
  public static class Options extends OptionsBase {
    @Option(
        name = "dex",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Generate dex output.")
    public boolean dex;

    @Option(
        name = "debug",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Print debug information.")
    public boolean debug;

    @Option(
        name = "verbose",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Print verbose information.")
    public boolean verbose;

    @Option(
        name = "positions",
        defaultValue = "lines",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "What source-position information to keep. One of: none, lines, important.")
    public String positions;

    @Option(
        name = "no-locals",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Don't keep local variable information.")
    public boolean noLocals;

    @Option(
        name = "statistics",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Print statistics information.")
    public boolean statistics;

    @Option(
        name = "no-optimize",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Don't optimize.")
    public boolean noOptimize;

    @Option(
        name = "optimize-list",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "File listing methods to optimize.")
    public String optimizeList;

    @Option(
        name = "no-optimize-list",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "File listing methods not to optimize.")
    public String noOptimizeList;

    @Option(
        name = "no-strict",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Disable strict file/class name checks.")
    public boolean noStrict;

    @Option(
        name = "keep-classes",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Keep input class files in in output jar.")
    public boolean keepClasses;

    @Option(
        name = "output",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "Output file or directory.")
    public String output;

    @Option(
        name = "dump-to",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "File to dump information to.")
    public String dumpTo;

    @Option(
        name = "dump-width",
        defaultValue = "8",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Max width for columns in dump output.")
    public int dumpWidth;

    @Option(
        name = "dump-method",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "Method to dump information for.")
    public String methodToDump;

    @Option(
        name = "dump",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Dump information.")
    public boolean dump;

    @Option(
        name = "verbose-dump",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Dump verbose information.")
    public boolean verboseDump;

    @Option(
        name = "no-files",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Don't fail if given no files.")
    public boolean noFiles;

    @Option(
        name = "core-library",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Construct a core library.")
    public boolean coreLibrary;

    @Option(
        name = "num-threads",
        defaultValue = "1",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Number of threads to run with.")
    public int numThreads;

    @Option(
        name = "incremental",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Merge result with the output if it exists.")
    public boolean incremental;

    @Option(
        name = "force-jumbo",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Force use of string-jumbo instructions.")
    public boolean forceJumbo;

    @Option(
        name = "no-warning",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Suppress warnings.")
    public boolean noWarning;

    @Option(
        name = "set-max-idx-number",
        defaultValue = "0",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Undocumented: Set maximal index number to use in a dex file.")
    public int maxIndexNumber;

    @Option(
        name = "minimal-main-dex",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Produce smallest possible main dex.")
    public boolean minimalMainDex;

    @Option(
        name = "main-dex-list",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        converter = StringConverter.class,
        help = "File listing classes that must be in the main dex file.")
    public String mainDexList;

    @Option(
        name = "multi-dex",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Allow generation of multi-dex.")
    public boolean multiDex;

    @Option(
        name = "min-sdk-version",
        defaultValue = "1",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Minimum Android API level compatibility.")
    public int minApiLevel;

    @Option(
        name = "desugar-backport-statics",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Backport additional Java 8 APIs.")
    public boolean backportStatics;

    @Option(
        name = "input-list",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = StringConverter.class,
        help = "File listing input files.")
    public String inputList;

    @Option(
        name = "version",
        defaultValue = "false", // dx's default
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        allowMultiple = false,
        help = "Print the version of this tool.")
    public boolean version;
  }

  /** Compatibility options parsing for the DX --dex sub-command. */
  public static class DxCompatOptions {

    // Final values after parsing.
    // Note: These are ordered by their occurrence in "dx --help"
    public final boolean help;
    public final boolean version;
    public final boolean debug;
    public final boolean verbose;
    public final PositionInfo positions;
    public final boolean noLocals;
    public final boolean noOptimize;
    public final boolean statistics;
    public final String optimizeList;
    public final String noOptimizeList;
    public final boolean noStrict;
    public final boolean keepClasses;
    public final String output;
    public final String dumpTo;
    public final int dumpWidth;
    public final String methodToDump;
    public final boolean verboseDump;
    public final boolean dump;
    public final boolean noFiles;
    public final boolean coreLibrary;
    public final int numThreads;
    public final boolean incremental;
    public final boolean forceJumbo;
    public final boolean noWarning;
    public final boolean multiDex;
    public final String mainDexList;
    public final boolean minimalMainDex;
    public final int minApiLevel;
    public final boolean backportStatics;
    public final String inputList;
    public final List<String> inputs;
    // Undocumented option
    public final int maxIndexNumber;

    /**
     * Values for dx --positions flag. Corresponding to "none", "important", "lines", "throwing".
     */
    public enum PositionInfo {
      NONE,
      IMPORTANT,
      LINES,
      THROWING
    }

    /** Exception thrown on invalid dx compat usage. */
    public static class DxUsageMessage extends Exception {

      public final String message;

      DxUsageMessage(String message) {
        this.message = message;
      }

      void printHelpOn(PrintStream sink) throws IOException {
        sink.println(message);
      }
    }

    private DxCompatOptions(Options options, List<String> remaining) {
      help = false;
      version = options.version;
      debug = options.debug;
      verbose = options.debug;
      switch (options.positions) {
        case "none":
          positions = PositionInfo.NONE;
          break;
        case "lines":
          positions = PositionInfo.LINES;
          break;
        case "throwing":
          positions = PositionInfo.THROWING;
          break;
        case "important":
          positions = PositionInfo.IMPORTANT;
          break;
        default:
          throw new AssertionError("Unreachable");
      }
      noLocals = options.noLocals;
      noOptimize = options.noOptimize;
      statistics = options.statistics;
      optimizeList = options.optimizeList;
      noOptimizeList = options.noOptimizeList;
      noStrict = options.noStrict;
      keepClasses = options.keepClasses;
      output = options.output;
      dumpTo = options.dumpTo;
      dumpWidth = options.dumpWidth;
      methodToDump = options.methodToDump;
      dump = options.dump;
      verboseDump = options.verboseDump;
      noFiles = options.noFiles;
      coreLibrary = options.coreLibrary;
      numThreads = options.numThreads;
      incremental = options.incremental;
      forceJumbo = options.forceJumbo;
      noWarning = options.noWarning;
      multiDex = options.multiDex;
      mainDexList = options.mainDexList;
      minimalMainDex = options.minimalMainDex;
      minApiLevel = options.minApiLevel;
      backportStatics = options.backportStatics;
      inputList = options.inputList;
      inputs = remaining;
      maxIndexNumber = options.maxIndexNumber;
    }

    public static DxCompatOptions parse(String[] args) {
      OptionsParser optionsParser = OptionsParser.builder().optionsClasses(Options.class).build();
      optionsParser.parseAndExitUponError(args);
      Options options = optionsParser.getOptions(Options.class);
      return new DxCompatOptions(options, optionsParser.getResidue());
    }
  }

  public static void main(String[] args) throws IOException {
    try {
      run(args);
    } catch (DxUsageMessage e) {
      System.err.println(USAGE_HEADER);
      e.printHelpOn(System.err);
      System.exit(1);
    } catch (CompilationFailedException e) {
      throw new AssertionError("Failure", e);
    }
  }

  private static void run(String[] args)
      throws DxUsageMessage, IOException, CompilationFailedException {
    DxCompatOptions dexArgs = DxCompatOptions.parse(args);
    if (dexArgs.version) {
      System.out.println("CompatDx " + Version.getVersionString());
      return;
    }
    CompilationMode mode = CompilationMode.RELEASE;
    Path output = null;
    List<Path> inputs = new ArrayList<>();
    boolean singleDexFile = !dexArgs.multiDex;
    Path mainDexList = null;
    int numberOfThreads = 1;

    for (String path : dexArgs.inputs) {
      processPath(Paths.get(path), inputs);
    }
    if (inputs.isEmpty()) {
      if (dexArgs.noFiles) {
        return;
      }
      throw new DxUsageMessage("No input files specified");
    }

    if (dexArgs.dump && dexArgs.verbose) {
      System.out.println("Warning: dump is not supported");
    }

    if (dexArgs.verboseDump) {
      throw new Unimplemented("verbose dump file not yet supported");
    }

    if (dexArgs.methodToDump != null) {
      throw new Unimplemented("method-dump not yet supported");
    }

    if (dexArgs.output != null) {
      output = Paths.get(dexArgs.output);
      if (FileUtils.isDexFile(output)) {
        if (!singleDexFile) {
          throw new DxUsageMessage("Cannot output to a single dex-file when running with multidex");
        }
      } else if (!FileUtils.isArchive(output)
          && (!output.toFile().exists() || !output.toFile().isDirectory())) {
        throw new DxUsageMessage(
            "Unsupported output file or output directory does not exist. "
                + "Output must be a directory or a file of type dex, apk, jar or zip.");
      }
    }

    if (dexArgs.dumpTo != null && dexArgs.verbose) {
      System.out.println("dump-to file not yet supported");
    }

    if (dexArgs.positions == PositionInfo.NONE && dexArgs.verbose) {
      System.out.println("Warning: no support for positions none.");
    }

    if (dexArgs.positions == PositionInfo.LINES && !dexArgs.noLocals) {
      mode = CompilationMode.DEBUG;
    }

    if (dexArgs.incremental) {
      throw new Unimplemented("incremental merge not supported yet");
    }

    if (dexArgs.forceJumbo && dexArgs.verbose) {
      System.out.println(
          "Warning: no support for forcing jumbo-strings.\n"
              + "Strings will only use jumbo-string indexing if necessary.\n"
              + "Make sure that any dex merger subsequently used "
              + "supports correct handling of jumbo-strings (eg, D8/R8 does).");
    }

    if (dexArgs.noOptimize && dexArgs.verbose) {
      System.out.println("Warning: no support for not optimizing");
    }

    if (dexArgs.optimizeList != null) {
      throw new Unimplemented("no support for optimize-method list");
    }

    if (dexArgs.noOptimizeList != null) {
      throw new Unimplemented("no support for dont-optimize-method list");
    }

    if (dexArgs.statistics && dexArgs.verbose) {
      System.out.println("Warning: no support for printing statistics");
    }

    if (dexArgs.numThreads > 1) {
      numberOfThreads = dexArgs.numThreads;
    }

    if (dexArgs.mainDexList != null) {
      mainDexList = Paths.get(dexArgs.mainDexList);
    }

    if (dexArgs.noStrict) {
      if (dexArgs.verbose) {
        System.out.println("Warning: conservative main-dex list not yet supported");
      }
    } else {
      if (dexArgs.verbose) {
        System.out.println("Warning: strict name checking not yet supported");
      }
    }

    if (dexArgs.minimalMainDex && dexArgs.verbose) {
      System.out.println("Warning: minimal main-dex support is not yet supported");
    }

    if (dexArgs.maxIndexNumber != 0 && dexArgs.verbose) {
      System.out.println("Warning: internal maximum-index setting is not supported");
    }

    if (numberOfThreads < 1) {
      throw new DxUsageMessage("Invalid numThreads value of " + numberOfThreads);
    }
    ExecutorService executor = Executors.newWorkStealingPool(numberOfThreads);

    try {
      D8Command.Builder builder = D8Command.builder();
      inputs.forEach(
          input ->
              builder.addProgramResourceProvider(ArchiveResourceProvider.fromArchive(input, true)));

      builder
          // .addProgramFiles(inputs)
          .setProgramConsumer(createConsumer(inputs, output, singleDexFile, dexArgs.keepClasses))
          .setMode(mode)
          .setDisableDesugaring(true) // DX does not desugar.
          .setMinApiLevel(dexArgs.minApiLevel);
      if (mainDexList != null) {
        builder.addMainDexListFiles(mainDexList);
      }
      if (dexArgs.backportStatics) {
        CompatDxSupport.enableDesugarBackportStatics(builder);
      }
      try {
        // Check if the referenced r8.jar has these methods. If so, the support code accessing
        // the internals is not required.
        Method setEnableMainDexListCheck =
            D8Command.Builder.class.getDeclaredMethod("setEnableMainDexListCheck", boolean.class);
        Method setMinimalMainDex =
            D8Command.Builder.class.getDeclaredMethod("setMinimalMainDex", boolean.class);
        // The methods are package private to not reveal them as part of the external API.
        setEnableMainDexListCheck.setAccessible(true);
        setMinimalMainDex.setAccessible(true);
        setEnableMainDexListCheck.invoke(builder, Boolean.FALSE);
        setMinimalMainDex.invoke(builder, dexArgs.minimalMainDex);
        D8.run(builder.build());
      } catch (ReflectiveOperationException e) {
        // Go through the support support code accessing the internals for the compilation.
        CompatDxSupport.run(builder.build(), dexArgs.minimalMainDex);
      }
    } finally {
      executor.shutdown();
    }
  }

  private static ProgramConsumer createConsumer(
      List<Path> inputs, Path output, boolean singleDexFile, boolean keepClasses)
      throws DxUsageMessage {
    if (output == null) {
      return DexIndexedConsumer.emptyConsumer();
    }
    if (singleDexFile) {
      return new SingleDexFileConsumer(
          FileUtils.isDexFile(output)
              ? new NamedDexFileConsumer(output)
              : createDexConsumer(output, inputs, keepClasses));
    }
    return createDexConsumer(output, inputs, keepClasses);
  }

  private static DexIndexedConsumer createDexConsumer(
      Path output, List<Path> inputs, boolean keepClasses) throws DxUsageMessage {
    if (keepClasses) {
      if (!FileUtils.isArchive(output)) {
        throw new DxCompatOptions.DxUsageMessage(
            "Output must be an archive when --keep-classes is set.");
      }
      return new ArchiveConsumer(output, inputs);
    }
    return FileUtils.isArchive(output)
        ? new ArchiveConsumer(output)
        : new DexIndexedConsumer.DirectoryConsumer(output);
  }

  private static class SingleDexFileConsumer extends DexIndexedConsumer.ForwardingConsumer {

    private byte[] bytes = null;

    public SingleDexFileConsumer(DexIndexedConsumer consumer) {
      super(consumer);
    }

    @Override
    public void accept(
        int fileIndex, ByteDataView data, Set<String> descriptors, DiagnosticsHandler handler) {
      if (fileIndex > 0) {
        throw new CompilationError(
            "Compilation result could not fit into a single dex file. "
                + "Reduce the input-program size or run with --multi-dex enabled");
      }
      verify(bytes == null, "Should not have been populated until now");
      // Store a copy of the bytes as we may not assume the backing is valid after accept returns.
      bytes = data.copyByteData();
    }

    @Override
    public void finished(DiagnosticsHandler handler) {
      if (bytes != null) {
        super.accept(0, ByteDataView.of(bytes), null, handler);
      }
      super.finished(handler);
    }
  }

  private static class NamedDexFileConsumer extends DexIndexedConsumer.ForwardingConsumer {

    private final Path output;

    public NamedDexFileConsumer(Path output) {
      super(null);
      this.output = output;
    }

    @Override
    public void accept(
        int fileIndex, ByteDataView data, Set<String> descriptors, DiagnosticsHandler handler) {
      StandardOpenOption[] options = {
        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING
      };
      try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(output, options))) {
        stream.write(data.getBuffer(), data.getOffset(), data.getLength());
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, new PathOrigin(output)));
      }
    }
  }

  /**
   * Consumer for writing the generated classes.dex files to an archive. Supports writing the input
   * class files to the same archive as well.
   */
  private static class ArchiveConsumer implements DexIndexedConsumer {
    private final Path path;
    private final List<Path> inputs;
    private final Origin origin;
    private ZipOutputStream stream;
    private int nextClassesDexIndex;
    private final Map<Integer, ClassesDexFileData> pendingClassesDexFiles = new HashMap<>();

    /** Content of a classes.dex file */
    private static class ClassesDexFileData {
      private final int index;
      private final ByteDataView content;

      private ClassesDexFileData(int index, ByteDataView content) {
        this.index = index;
        this.content = content;
      }
    }

    ArchiveConsumer(Path path) {
      this(path, ImmutableList.of());
    }

    ArchiveConsumer(Path path, List<Path> inputs) {
      this.path = path;
      this.inputs = inputs;
      this.origin = new PathOrigin(path);
    }

    @Override
    public void accept(
        int fileIndex, ByteDataView data, Set<String> descriptors, DiagnosticsHandler handler) {
      ensureOpenArchive(handler);
      addIndexedClassesDexFile(fileIndex, data, handler);
    }

    @Override
    public void finished(DiagnosticsHandler handler) {
      verify(pendingClassesDexFiles.isEmpty(), "All DEX files should have been written");
      if (stream == null) {
        return;
      }
      try {
        writeInputClassesToArchive(handler);
        stream.close();
        stream = null;
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, origin));
      }
    }

    private synchronized void addIndexedClassesDexFile(
        int fileIndex, ByteDataView data, DiagnosticsHandler handler) {
      // Always add the classes.dex files in <code>fileIndex</code> order to have stable output.
      // Store the ones which arrive out-of-order and write as soon as possible.
      pendingClassesDexFiles.put(
          fileIndex, new ClassesDexFileData(fileIndex, ByteDataView.of(data.copyByteData())));
      while (pendingClassesDexFiles.containsKey(nextClassesDexIndex)) {
        ClassesDexFileData classesDexFileData = pendingClassesDexFiles.get(nextClassesDexIndex);
        writeClassesDexFile(classesDexFileData, handler);
        pendingClassesDexFiles.remove(nextClassesDexIndex);
        nextClassesDexIndex++;
      }
    }

    /** Get or open the zip output stream. */
    private synchronized void ensureOpenArchive(DiagnosticsHandler handler) {
      if (stream != null) {
        return;
      }
      try {
        stream =
            new ZipOutputStream(
                Files.newOutputStream(
                    path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING));
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, origin));
      }
    }

    private void writeClassesDexFile(
        ClassesDexFileData classesDexFileData, DiagnosticsHandler handler) {
      try {
        ZipUtils.writeToZipStream(
            getDexFileName(classesDexFileData.index),
            classesDexFileData.content,
            ZipEntry.DEFLATED,
            stream);
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, origin));
      }
    }

    protected String getDexFileName(int fileIndex) {
      return "classes" + (fileIndex == 0 ? "" : (fileIndex + 1)) + FileUtils.DEX_EXTENSION;
    }

    private void writeClassFile(String name, ByteDataView content, DiagnosticsHandler handler) {
      try {
        ZipUtils.writeToZipStream(name, content, ZipEntry.DEFLATED, stream);
      } catch (IOException e) {
        handler.error(new ExceptionDiagnostic(e, origin));
      }
    }

    @SuppressWarnings("JdkObsolete") // Uses Enumeration by design.
    private void writeInputClassesToArchive(DiagnosticsHandler handler) throws IOException {
      // For each input archive file, add all class files within.
      for (Path input : inputs) {
        if (FileUtils.isArchive(input)) {
          try (ZipFile zipFile = new ZipFile(input.toFile(), UTF_8)) {
            final Enumeration<? extends ZipEntry> entries = zipFile.entries();
            while (entries.hasMoreElements()) {
              ZipEntry entry = entries.nextElement();
              if (FileUtils.isClassFile(entry.getName())) {
                try (InputStream entryStream = zipFile.getInputStream(entry)) {
                  byte[] bytes = ByteStreams.toByteArray(entryStream);
                  writeClassFile(entry.getName(), ByteDataView.of(bytes), handler);
                }
              }
            }
          }
        }
      }
    }
  }

  private static void processPath(Path path, List<Path> files) throws IOException {
    if (!Files.exists(path)) {
      throw new CompilationError("File does not exist: " + path);
    }
    if (Files.isDirectory(path)) {
      processDirectory(path, files);
      return;
    }
    if (FileUtils.isZipFile(path) || FileUtils.isJarFile(path) || FileUtils.isClassFile(path)) {
      files.add(path);
      return;
    }
    if (FileUtils.isApkFile(path)) {
      throw new Unimplemented("apk files not yet supported: " + path);
    }
  }

  private static void processDirectory(Path directory, List<Path> files) throws IOException {
    verify(Files.exists(directory), "Directory must exist");

    try (Stream<Path> pathStream = Files.list(directory)) {
      for (Path file : pathStream.collect(toList())) {
        processPath(file, files);
      }
    }
  }

  private CompatDx() {}
}
