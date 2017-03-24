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
package com.google.devtools.build.android;

import static java.nio.charset.StandardCharsets.UTF_8;

import android.databinding.AndroidDataBinding;
import android.databinding.cli.ProcessXmlOptions;
import com.android.annotations.NonNull;
import com.android.annotations.Nullable;
import com.android.builder.core.VariantConfiguration;
import com.android.builder.core.VariantType;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.builder.internal.SymbolLoader;
import com.android.builder.internal.SymbolWriter;
import com.android.builder.model.AaptOptions;
import com.android.ide.common.internal.CommandLineRunner;
import com.android.ide.common.internal.ExecutorSingleton;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.io.FileWrapper;
import com.android.io.StreamException;
import com.android.repository.Revision;
import com.android.utils.ILogger;
import com.android.utils.StdLogger;
import com.android.xml.AndroidManifest;
import com.google.common.base.Joiner;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.RevisionConverter;
import com.google.devtools.build.android.SplitConfigurationFilter.UnrecognizedSplitsException;
import com.google.devtools.build.android.resources.RClassGenerator;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.logging.Logger;
import javax.xml.xpath.XPathExpressionException;

/**
 * Provides a wrapper around the AOSP build tools for resource processing.
 */
public class AndroidResourceProcessor {
  static final Logger logger = Logger.getLogger(AndroidResourceProcessor.class.getName());

  /**
   * Options class containing flags for Aapt setup.
   */
  public static final class AaptConfigOptions extends OptionsBase {
    @Option(name = "buildToolsVersion",
        defaultValue = "null",
        converter = RevisionConverter.class,
        category = "config",
        help = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2")
    public Revision buildToolsVersion;

    @Option(name = "aapt",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Aapt tool location for resource packaging.")
    public Path aapt;

    @Option(name = "featureOf",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "config",
        help = "Base apk path.")
    public Path featureOf;

    @Option(name = "featureAfter",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "config",
        help = "Apk path of previous split (if any).")
    public Path featureAfter;

    @Option(name = "annotationJar",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Annotation Jar for builder invocations.")
    public Path annotationJar;

    @Option(name = "androidJar",
        defaultValue = "null",
        converter = ExistingPathConverter.class,
        category = "tool",
        help = "Path to the android jar for resource packaging and building apks.")
    public Path androidJar;

    @Option(name = "useAaptCruncher",
        defaultValue = "auto",
        category = "config",
        help = "Use the legacy aapt cruncher, defaults to true for non-LIBRARY packageTypes. "
            + " LIBRARY packages do not benefit from the additional processing as the resources"
            + " will need to be reprocessed during the generation of the final apk. See"
            + " https://code.google.com/p/android/issues/detail?id=67525 for a discussion of the"
            + " different png crunching methods.")
    public TriState useAaptCruncher;

    @Option(name = "uncompressedExtensions",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of file extensions not to compress.")
    public List<String> uncompressedExtensions;

    @Option(name = "assetsToIgnore",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of assets extensions to ignore.")
    public List<String> assetsToIgnore;

    @Option(name = "debug",
        defaultValue = "false",
        category = "config",
        help = "Indicates if it is a debug build.")
    public boolean debug;

    @Option(name = "resourceConfigs",
        defaultValue = "",
        converter = CommaSeparatedOptionListConverter.class,
        category = "config",
        help = "A list of resource config filters to pass to aapt.")
    public List<String> resourceConfigs;

    private static final String ANDROID_SPLIT_DOCUMENTATION_URL =
        "https://developer.android.com/guide/topics/resources/providing-resources.html"
        + "#QualifierRules";

    @Option(
      name = "split",
      defaultValue = "required but ignored due to allowMultiple",
      category = "config",
      allowMultiple = true,
      help =
          "An individual split configuration to pass to aapt."
              + " Each split is a list of configuration filters separated by commas."
              + " Configuration filters are lists of configuration qualifiers separated by dashes,"
              + " as used in resource directory names and described on the Android developer site: "
              + ANDROID_SPLIT_DOCUMENTATION_URL
              + " For example, a split might be 'en-television,en-xxhdpi', containing English"
              + " assets which either are for TV screens or are extra extra high resolution."
              + " Multiple splits can be specified by passing this flag multiple times."
              + " Each split flag will produce an additional output file, named by replacing the"
              + " commas in the split specification with underscores, and appending the result to"
              + " the output package name following an underscore."
    )
    public List<String> splits;
  }

  /**
   * {@link AaptOptions} backed by an {@link AaptConfigOptions}.
   */
  public static final class FlagAaptOptions implements AaptOptions {
    private final AaptConfigOptions options;

    public FlagAaptOptions(AaptConfigOptions options) {
      this.options = options;
    }

    @Override
    public Collection<String> getNoCompress() {
      if (!options.uncompressedExtensions.isEmpty()) {
        return options.uncompressedExtensions;
      }
      return ImmutableList.of();
    }

    @Override
    public String getIgnoreAssets() {
      if (!options.assetsToIgnore.isEmpty()) {
        return Joiner.on(":").join(options.assetsToIgnore);
      }
      return null;
    }

    @Override
    public boolean getFailOnMissingConfigEntry() {
      return false;
    }

    @Override
    public List<String> getAdditionalParameters() {
      List<String> params = new java.util.ArrayList<String>();
      if (options.featureOf != null) {
         params.add("--feature-of");
         params.add(options.featureOf.toString());
      }
      if (options.featureAfter != null) {
         params.add("--feature-after");
         params.add(options.featureAfter.toString());
      }
      return ImmutableList.copyOf(params);
    }

  }

  private final StdLogger stdLogger;

  public AndroidResourceProcessor(StdLogger stdLogger) {
    this.stdLogger = stdLogger;
  }

  // TODO(bazel-team): Clean up this method call -- 13 params is too many.
  /** Processes resources for generated sources, configs and packaging resources. */
  public void processResources(
      Path aapt,
      Path androidJar,
      @Nullable Revision buildToolsVersion,
      VariantType variantType,
      boolean debug,
      String customPackageForR,
      AaptOptions aaptOptions,
      Collection<String> resourceConfigs,
      Collection<String> splits,
      MergedAndroidData primaryData,
      List<DependencyAndroidData> dependencyData,
      @Nullable Path sourceOut,
      @Nullable Path packageOut,
      @Nullable Path proguardOut,
      @Nullable Path mainDexProguardOut,
      @Nullable Path publicResourcesOut,
      @Nullable Path dataBindingInfoOut)
      throws IOException, InterruptedException, LoggedErrorException, UnrecognizedSplitsException {
    Path androidManifest = primaryData.getManifest();
    final Path resourceDir = processDataBindings(primaryData.getResourceDir(), dataBindingInfoOut,
        variantType, customPackageForR, androidManifest);

    final Path assetsDir = primaryData.getAssetDir();
    if (publicResourcesOut != null) {
      prepareOutputPath(publicResourcesOut.getParent());
    }
    runAapt(aapt,
        androidJar,
        buildToolsVersion,
        variantType,
        debug,
        customPackageForR,
        aaptOptions,
        resourceConfigs,
        splits,
        androidManifest,
        resourceDir,
        assetsDir,
        sourceOut,
        packageOut,
        proguardOut,
        mainDexProguardOut,
        publicResourcesOut);
    // The R needs to be created for each library in the dependencies,
    // but only if the current project is not a library.
    if (sourceOut != null && variantType != VariantType.LIBRARY) {
      writeDependencyPackageRJavaFiles(
          dependencyData, customPackageForR, androidManifest, sourceOut);
    }
    // Reset the output date stamps.
    if (proguardOut != null) {
      Files.setLastModifiedTime(proguardOut, FileTime.fromMillis(0L));
    }
    if (mainDexProguardOut != null) {
      Files.setLastModifiedTime(mainDexProguardOut, FileTime.fromMillis(0L));
    }
    if (packageOut != null) {
      Files.setLastModifiedTime(packageOut, FileTime.fromMillis(0L));
      if (!splits.isEmpty()) {
        Iterable<Path> splitFilenames = findAndRenameSplitPackages(packageOut, splits);
        for (Path splitFilename : splitFilenames) {
          Files.setLastModifiedTime(splitFilename, FileTime.fromMillis(0L));
        }
      }
    }
    if (publicResourcesOut != null && Files.exists(publicResourcesOut)) {
      Files.setLastModifiedTime(publicResourcesOut, FileTime.fromMillis(0L));
    }
  }

  public void runAapt(
      Path aapt,
      Path androidJar,
      @Nullable Revision buildToolsVersion,
      VariantType variantType,
      boolean debug,
      String customPackageForR,
      AaptOptions aaptOptions,
      Collection<String> resourceConfigs,
      Collection<String> splits,
      Path androidManifest,
      Path resourceDir,
      Path assetsDir,
      Path sourceOut,
      @Nullable Path packageOut,
      @Nullable Path proguardOut,
      @Nullable Path mainDexProguardOut,
      @Nullable Path publicResourcesOut)
      throws InterruptedException, LoggedErrorException, IOException {
    AaptCommandBuilder commandBuilder =
        new AaptCommandBuilder(aapt)
        .forBuildToolsVersion(buildToolsVersion)
        .forVariantType(variantType)
        // first argument is the command to be executed, "package"
        .add("package")
        // If the logger is verbose, set aapt to be verbose
        .when(stdLogger.getLevel() == StdLogger.Level.VERBOSE).thenAdd("-v")
        // Overwrite existing files, if they exist.
        .add("-f")
        // Resources are precrunched in the merge process.
        .add("--no-crunch")
        // Do not automatically generate versioned copies of vector XML resources.
        .whenVersionIsAtLeast(new Revision(23)).thenAdd("--no-version-vectors")
        // Add the android.jar as a base input.
        .add("-I", androidJar)
        // Add the manifest for validation.
        .add("-M", androidManifest.toAbsolutePath())
        // Maybe add the resources if they exist
        .when(Files.isDirectory(resourceDir)).thenAdd("-S", resourceDir)
        // Maybe add the assets if they exist
        .when(Files.isDirectory(assetsDir)).thenAdd("-A", assetsDir)
        // Outputs
        .when(sourceOut != null).thenAdd("-m")
        .add("-J", prepareOutputPath(sourceOut))
        .add("--output-text-symbols", prepareOutputPath(sourceOut))
        .add("-F", packageOut)
        .add("-G", proguardOut)
        .whenVersionIsAtLeast(new Revision(24)).thenAdd("-D", mainDexProguardOut)
        .add("-P", publicResourcesOut)
        .when(debug).thenAdd("--debug-mode")
        .add("--custom-package", customPackageForR)
        // If it is a library, do not generate final java ids.
        .whenVariantIs(VariantType.LIBRARY).thenAdd("--non-constant-id")
        .add("--ignore-assets", aaptOptions.getIgnoreAssets())
        .when(aaptOptions.getFailOnMissingConfigEntry()).thenAdd("--error-on-missing-config-entry")
        // Never compress apks.
        .add("-0", "apk")
        // Add custom no-compress extensions.
        .addRepeated("-0", aaptOptions.getNoCompress())
        // Filter by resource configuration type.
        .add("-c", Joiner.on(',').join(resourceConfigs))
        // Split APKs if any splits were specified.
        .whenVersionIsAtLeast(new Revision(23)).thenAddRepeated("--split", splits);
    for (String additional : aaptOptions.getAdditionalParameters()) {
      commandBuilder.add(additional);
    }
    try {
      new CommandLineRunner(stdLogger).runCmdLine(commandBuilder.build(), null);
    } catch (LoggedErrorException e) {
      // Add context and throw the error to resume processing.
      throw new LoggedErrorException(
          e.getCmdLineError(), getOutputWithSourceContext(aapt, e.getOutput()), e.getCmdLine());
    }
  }

  /** Adds 10 lines of source to each syntax error. Very useful for debugging. */
  private List<String> getOutputWithSourceContext(Path aapt, List<String> lines)
      throws IOException {
    List<String> outputWithSourceContext = new ArrayList<>();
    for (String line : lines) {
      if (line.contains("Duplicate file") || line.contains("Original")) {
        String[] parts = line.split(":");
        String fileName = parts[0].trim();
        outputWithSourceContext.add("\n" + fileName + ":\n\t");
        outputWithSourceContext.add(
            Joiner.on("\n\t")
                .join(
                    Files.readAllLines(
                        aapt.getFileSystem().getPath(fileName), StandardCharsets.UTF_8)));
      } else if (line.contains("error")) {
        String[] parts = line.split(":");
        String fileName = parts[0].trim();
        try {
          int lineNumber = Integer.valueOf(parts[1].trim());
          StringBuilder expandedError =
              new StringBuilder("\nError at " + lineNumber + " : " + line);
          List<String> errorSource =
              Files.readAllLines(aapt.getFileSystem().getPath(fileName), StandardCharsets.UTF_8);
          for (int i = Math.max(lineNumber - 5, 0);
              i < Math.min(lineNumber + 5, errorSource.size());
              i++) {
            expandedError.append("\n").append(i).append("\t:  ").append(errorSource.get(i));
          }
          outputWithSourceContext.add(expandedError.toString());
        } catch (IOException | NumberFormatException formatError) {
          outputWithSourceContext.add("error parsing line" + line);
          stdLogger.error(formatError, "error during reading source %s", fileName);
        }
      } else {
        outputWithSourceContext.add(line);
      }
    }
    return outputWithSourceContext;
  }

  /**
   * If resources exist and a data binding layout info file is requested: processes data binding
   * declarations over those resources, populates the output file, and creates a new resources
   * directory with data binding expressions stripped out (so aapt, which doesn't understand
   * data binding, can properly read them).
   *
   * <p>Returns the resources directory that aapt should read.
   */
  static Path processDataBindings(Path resourceDir, Path dataBindingInfoOut,
      VariantType variantType, String packagePath, Path androidManifest)
      throws IOException {

    if (dataBindingInfoOut == null) {
      return resourceDir;
    } else if (!Files.isDirectory(resourceDir)) {
      // No resources: no data binding needed. Create a dummy file to satisfy declared outputs.
      Files.createFile(dataBindingInfoOut);
      return resourceDir;
    }

    // Strip the file name (the data binding library automatically adds it back in).
    // ** The data binding library assumes this file is called "layout-info.zip". **
    dataBindingInfoOut = dataBindingInfoOut.getParent();
    if (Files.notExists(dataBindingInfoOut)) {
      Files.createDirectory(dataBindingInfoOut);
    }

    Path processedResourceDir = resourceDir.resolveSibling("res_without_databindings");
    if (Files.notExists(processedResourceDir)) {
      Files.createDirectory(processedResourceDir);
    }

    ProcessXmlOptions options = new ProcessXmlOptions();
    options.setAppId(packagePath);
    options.setLibrary(variantType == VariantType.LIBRARY);
    options.setResInput(resourceDir.toFile());
    options.setResOutput(processedResourceDir.toFile());
    options.setLayoutInfoOutput(dataBindingInfoOut.toFile());
    options.setZipLayoutInfo(true); // Aggregate data-bound .xml files into a single .zip.

    try {
      Object minSdk = AndroidManifest.getMinSdkVersion(new FileWrapper(androidManifest.toFile()));
      if (minSdk instanceof Integer) {
        options.setMinSdk(((Integer) minSdk).intValue());
      } else {
        // TODO(bazel-team): Enforce the minimum SDK check.
        options.setMinSdk(15);
      }
    } catch (XPathExpressionException | StreamException e) {
      // TODO(bazel-team): Enforce the minimum SDK check.
      options.setMinSdk(15);
    }

    try {
      AndroidDataBinding.doRun(options);
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
    return processedResourceDir;
  }

  /** Task to parse java package from AndroidManifest.xml */
  private static final class PackageParsingTask implements Callable<String> {

    private final File manifest;

    PackageParsingTask(File manifest) {
      this.manifest = manifest;
    }

    @Override
    public String call() throws Exception {
      return VariantConfiguration.getManifestPackage(manifest);
    }
  }

  /** Task to load and parse R.txt symbols */
  private static final class SymbolLoadingTask implements Callable<Object> {

    private final SymbolLoader symbolLoader;

    SymbolLoadingTask(SymbolLoader symbolLoader) {
      this.symbolLoader = symbolLoader;
    }

    @Override
    public Object call() throws Exception {
      symbolLoader.load();
      return null;
    }
  }

  @Nullable
  public SymbolLoader loadResourceSymbolTable(
      List<SymbolFileProvider> libraries,
      String appPackageName,
      Path primaryRTxt,
      Multimap<String, SymbolLoader> libMap) throws IOException {
    // The reported availableProcessors may be higher than the actual resources
    // (on a shared system). On the other hand, a lot of the work is I/O, so it's not completely
    // CPU bound. As a compromise, divide by 2 the reported availableProcessors.
    int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
    ListeningExecutorService executorService = MoreExecutors.listeningDecorator(
        Executors.newFixedThreadPool(numThreads));
    try (Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
      // Load the package names from the manifest files.
      Map<SymbolFileProvider, ListenableFuture<String>> packageJobs = new HashMap<>();
      for (final SymbolFileProvider lib : libraries) {
        packageJobs.put(lib, executorService.submit(new PackageParsingTask(lib.getManifest())));
      }
      Map<SymbolFileProvider, String> packageNames = new HashMap<>();
      try {
        for (Map.Entry<SymbolFileProvider, ListenableFuture<String>> entry : packageJobs
            .entrySet()) {
          packageNames.put(entry.getKey(), entry.getValue().get());
        }
      } catch (InterruptedException | ExecutionException e) {
        throw new IOException("Failed to load package name: ", e);
      }
      // Associate the packages with symbol files.
      for (SymbolFileProvider lib : libraries) {
        String packageName = packageNames.get(lib);
        // If the library package matches the app package skip -- the final app resource IDs are
        // stored in the primaryRTxt file.
        if (appPackageName.equals(packageName)) {
          continue;
        }
        File rFile = lib.getSymbolFile();
        // If the library has no resource, this file won't exist.
        if (rFile.isFile()) {
          SymbolLoader libSymbols = new SymbolLoader(rFile, stdLogger);
          libMap.put(packageName, libSymbols);
        }
      }
      // Even if there are no libraries, load fullSymbolValues, in case we only have resources
      // defined for the binary.
      File primaryRTxtFile = primaryRTxt.toFile();
      SymbolLoader fullSymbolValues = null;
      if (primaryRTxtFile.isFile()) {
        fullSymbolValues = new SymbolLoader(primaryRTxtFile, stdLogger);
      }
      // Now load the symbol files in parallel.
      List<ListenableFuture<?>> loadJobs = new ArrayList<>();
      Iterable<SymbolLoader> toLoad = fullSymbolValues != null
          ? Iterables.concat(libMap.values(), ImmutableList.of(fullSymbolValues))
          : libMap.values();
      for (final SymbolLoader loader : toLoad) {
        loadJobs.add(executorService.submit(new SymbolLoadingTask(loader)));
      }
      try {
        Futures.allAsList(loadJobs).get();
      } catch (InterruptedException | ExecutionException e) {
        throw new IOException("Failed to load SymbolFile: ", e);
      }
      return fullSymbolValues;
    }
  }

  void writeDependencyPackageRJavaFiles(
      List<DependencyAndroidData> dependencyData,
      String customPackageForR,
      Path androidManifest,
      Path sourceOut) throws IOException {
    List<SymbolFileProvider> libraries = new ArrayList<>();
    for (DependencyAndroidData dataDep : dependencyData) {
      SymbolFileProvider library = dataDep.asSymbolFileProvider();
      libraries.add(library);
    }
    String appPackageName = customPackageForR;
    if (appPackageName == null) {
      appPackageName = VariantConfiguration.getManifestPackage(androidManifest.toFile());
    }
    Multimap<String, SymbolLoader> libSymbolMap = ArrayListMultimap.create();
    Path primaryRTxt = sourceOut != null ? sourceOut.resolve("R.txt") : null;
    if (primaryRTxt != null && !libraries.isEmpty()) {
      SymbolLoader fullSymbolValues = loadResourceSymbolTable(libraries,
          appPackageName, primaryRTxt, libSymbolMap);
      if (fullSymbolValues != null) {
        writePackageRJavaFiles(libSymbolMap, fullSymbolValues, sourceOut);
      }
    }
  }

  private void writePackageRJavaFiles(
      Multimap<String, SymbolLoader> libMap,
      SymbolLoader fullSymbolValues,
      Path sourceOut) throws IOException {
    // Loop on all the package name, merge all the symbols to write, and write.
    for (String packageName : libMap.keySet()) {
      Collection<SymbolLoader> symbols = libMap.get(packageName);
      SymbolWriter writer = new SymbolWriter(sourceOut.toString(), packageName, fullSymbolValues);
      for (SymbolLoader symbolLoader : symbols) {
        writer.addSymbolsToWrite(symbolLoader);
      }
      writer.write();
    }
  }

  void writePackageRClasses(
      Multimap<String, SymbolLoader> libMap,
      SymbolLoader fullSymbolValues,
      String appPackageName,
      Path classesOut,
      boolean finalFields) throws IOException {
    for (String packageName : libMap.keySet()) {
      Collection<SymbolLoader> symbols = libMap.get(packageName);
      RClassGenerator classWriter = RClassGenerator.fromSymbols(
          classesOut, packageName, fullSymbolValues, symbols, finalFields);
      classWriter.write();
    }
    // Unlike the R.java generation, we also write the app's R.class file so that the class
    // jar file can be complete (aapt doesn't generate it for us).
    RClassGenerator classWriter = RClassGenerator.fromSymbols(classesOut, appPackageName,
        fullSymbolValues, ImmutableList.of(fullSymbolValues), finalFields);
    classWriter.write();
  }

  /** Finds aapt's split outputs and renames them according to the input flags. */
  private Iterable<Path> findAndRenameSplitPackages(Path packageOut, Iterable<String> splits)
      throws UnrecognizedSplitsException, IOException {
    String prefix = packageOut.getFileName().toString() + "_";
    // The regex java string literal below is received as [\\{}\[\]*?] by the regex engine,
    // which produces a character class containing \{}[]*?
    // The replacement string literal is received as \\$0 by the regex engine, which places
    // a backslash before the match.
    String prefixGlob = prefix.replaceAll("[\\\\{}\\[\\]*?]", "\\\\$0") + "*";
    Path outputDirectory = packageOut.getParent();
    ImmutableList.Builder<String> filenameSuffixes = new ImmutableList.Builder<>();
    try (DirectoryStream<Path> glob = Files.newDirectoryStream(outputDirectory, prefixGlob)) {
      for (Path file : glob) {
        filenameSuffixes.add(file.getFileName().toString().substring(prefix.length()));
      }
    }
    Map<String, String> outputs =
        SplitConfigurationFilter.mapFilenamesToSplitFlags(filenameSuffixes.build(), splits);
    ImmutableList.Builder<Path> outputPaths = new ImmutableList.Builder<>();
    for (Map.Entry<String, String> splitMapping : outputs.entrySet()) {
      Path resultPath = packageOut.resolveSibling(prefix + splitMapping.getValue());
      outputPaths.add(resultPath);
      if (!splitMapping.getKey().equals(splitMapping.getValue())) {
        Path sourcePath = packageOut.resolveSibling(prefix + splitMapping.getKey());
        Files.move(sourcePath, resultPath);
      }
    }
    return outputPaths.build();
  }

  /** A logger that will print messages to a target OutputStream. */
  static final class PrintStreamLogger implements ILogger {
    private final PrintStream out;

    public PrintStreamLogger(PrintStream stream) {
      this.out = stream;
    }

    @Override
    public void error(@Nullable Throwable t, @Nullable String msgFormat, Object... args) {
      if (msgFormat != null) {
        out.println(String.format("Error: " + msgFormat, args));
      }
      if (t != null) {
        out.printf("Error: %s%n", t.getMessage());
      }
    }

    @Override
    public void warning(@NonNull String msgFormat, Object... args) {
      out.println(String.format("Warning: " + msgFormat, args));
    }

    @Override
    public void info(@NonNull String msgFormat, Object... args) {
      out.println(String.format("Info: " + msgFormat, args));
    }

    @Override
    public void verbose(@NonNull String msgFormat, Object... args) {
      out.println(String.format(msgFormat, args));
    }
  }

  public void writeDummyManifestForAapt(Path dummyManifest, String packageForR) throws IOException {
    Files.createDirectories(dummyManifest.getParent());
    Files.write(dummyManifest, String.format(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            + "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\""
            + " package=\"%s\">"
            + "</manifest>", packageForR).getBytes(UTF_8));
  }

  /**
   * Shutdown AOSP utilized thread-pool.
   */
  public void shutdown() {
    FullyQualifiedName.logCacheUsage(logger);
    // AOSP code never shuts down its singleton executor and leaves the process hanging.
    ExecutorSingleton.getExecutor().shutdownNow();
  }

  @Nullable
  private Path prepareOutputPath(@Nullable Path out) throws IOException {
    if (out == null) {
      return null;
    }
    return Files.createDirectories(out);
  }
}
