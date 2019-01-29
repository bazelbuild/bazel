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

import android.databinding.AndroidDataBinding;
import android.databinding.cli.ProcessXmlOptions;
import com.android.annotations.NonNull;
import com.android.annotations.Nullable;
import com.android.builder.core.VariantConfiguration;
import com.android.builder.core.VariantType;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.builder.model.AaptOptions;
import com.android.ide.common.internal.CommandLineRunner;
import com.android.ide.common.internal.ExecutorSingleton;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.repository.Revision;
import com.android.utils.ILogger;
import com.android.utils.StdLogger;
import com.google.common.base.Joiner;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.RevisionConverter;
import com.google.devtools.build.android.SplitConfigurationFilter.UnrecognizedSplitsException;
import com.google.devtools.build.android.junctions.JunctionCreator;
import com.google.devtools.build.android.junctions.NoopJunctionCreator;
import com.google.devtools.build.android.junctions.WindowsJunctionCreator;
import com.google.devtools.build.android.resources.ResourceSymbols;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.io.Closeable;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/** Provides a wrapper around the AOSP build tools for resource processing. */
public class AndroidResourceProcessor {
  static final Logger logger = Logger.getLogger(AndroidResourceProcessor.class.getName());

  /** Options class containing flags for Aapt setup. */
  public static final class AaptConfigOptions extends OptionsBase {
    @Option(
      name = "buildToolsVersion",
      defaultValue = "null",
      converter = RevisionConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2"
    )
    public Revision buildToolsVersion;

    @Option(
      name = "aapt",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "tool",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Aapt tool location for resource packaging."
    )
    public Path aapt;

    @Option(
      name = "featureOf",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Base apk path."
    )
    public Path featureOf;

    @Option(
      name = "featureAfter",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Apk path of previous split (if any)."
    )
    public Path featureAfter;

    @Option(
      name = "androidJar",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "tool",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the android jar for resource packaging and building apks."
    )
    public Path androidJar;

    @Option(
      name = "useAaptCruncher",
      defaultValue = "auto",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Use the legacy aapt cruncher, defaults to true for non-LIBRARY packageTypes. "
              + " LIBRARY packages do not benefit from the additional processing as the resources"
              + " will need to be reprocessed during the generation of the final apk. See"
              + " https://code.google.com/p/android/issues/detail?id=67525 for a discussion of the"
              + " different png crunching methods."
    )
    public TriState useAaptCruncher;

    @Option(
      name = "uncompressedExtensions",
      defaultValue = "",
      converter = CommaSeparatedOptionListConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A list of file extensions not to compress."
    )
    public List<String> uncompressedExtensions;

    @Option(
      name = "assetsToIgnore",
      defaultValue = "",
      converter = CommaSeparatedOptionListConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A list of assets extensions to ignore."
    )
    public List<String> assetsToIgnore;

    @Option(
      name = "debug",
      defaultValue = "false",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Indicates if it is a debug build."
    )
    public boolean debug;

    @Option(
      name = "resourceConfigs",
      defaultValue = "",
      converter = CommaSeparatedOptionListConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A list of resource config filters to pass to aapt."
    )
    public List<String> resourceConfigs;

    private static final String ANDROID_SPLIT_DOCUMENTATION_URL =
        "https://developer.android.com/guide/topics/resources/providing-resources.html"
            + "#QualifierRules";

    @Option(
      name = "split",
      defaultValue = "required but ignored due to allowMultiple",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
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

  /** {@link AaptOptions} backed by an {@link AaptConfigOptions}. */
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
  /**
   * Processes resources for generated sources, configs and packaging resources.
   *
   * <p>Returns a post-processed MergedAndroidData. Notably, the resources will be stripped of any
   * databinding expressions.
   */
  public MergedAndroidData processResources(
      Path tempRoot,
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
    final Path resourceDir =
        processDataBindings(
            primaryData.getResourceDir().resolveSibling("res_no_binding"),
            primaryData.getResourceDir(),
            dataBindingInfoOut,
            customPackageForR,
            /* shouldZipDataBindingInfo= */ true);

    final Path assetsDir = primaryData.getAssetDir();
    if (publicResourcesOut != null) {
      prepareOutputPath(publicResourcesOut.getParent());
    }
    runAapt(
        tempRoot,
        aapt,
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
    if (packageOut != null) {
      if (!splits.isEmpty()) {
        renameSplitPackages(packageOut, splits);
      }
    }
    return new MergedAndroidData(resourceDir, assetsDir, androidManifest);
  }

  public void runAapt(
      Path tempRoot,
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
    try (JunctionCreator junctions =
        System.getProperty("os.name").toLowerCase().startsWith("windows")
            ? new WindowsJunctionCreator(Files.createDirectories(tempRoot.resolve("juncts")))
            : new NoopJunctionCreator()) {
      sourceOut = junctions.create(sourceOut);
      AaptCommandBuilder commandBuilder =
          new AaptCommandBuilder(junctions.create(aapt))
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(variantType)
              // first argument is the command to be executed, "package"
              .add("package")
              // If the logger is verbose, set aapt to be verbose
              .when(stdLogger.getLevel() == StdLogger.Level.VERBOSE)
              .thenAdd("-v")
              // Overwrite existing files, if they exist.
              .add("-f")
              // Resources are precrunched in the merge process.
              .add("--no-crunch")
              // Do not automatically generate versioned copies of vector XML resources.
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              // Add the android.jar as a base input.
              .add("-I", junctions.create(androidJar))
              // Add the manifest for validation.
              .add("-M", junctions.create(androidManifest.toAbsolutePath()))
              // Maybe add the resources if they exist
              .when(Files.isDirectory(resourceDir))
              .thenAdd("-S", junctions.create(resourceDir))
              // Maybe add the assets if they exist
              .when(Files.isDirectory(assetsDir))
              .thenAdd("-A", junctions.create(assetsDir))
              // Outputs
              .when(sourceOut != null)
              .thenAdd("-m")
              .add("-J", prepareOutputPath(sourceOut))
              .add("--output-text-symbols", prepareOutputPath(sourceOut))
              .add("-F", junctions.create(packageOut))
              .add("-G", junctions.create(proguardOut))
              .whenVersionIsAtLeast(new Revision(24))
              .thenAdd("-D", junctions.create(mainDexProguardOut))
              .add("-P", junctions.create(publicResourcesOut))
              .when(debug)
              .thenAdd("--debug-mode")
              .add("--custom-package", customPackageForR)
              // If it is a library, do not generate final java ids.
              .whenVariantIs(VariantType.LIBRARY)
              .thenAdd("--non-constant-id")
              .add("--ignore-assets", aaptOptions.getIgnoreAssets())
              .when(aaptOptions.getFailOnMissingConfigEntry())
              .thenAdd("--error-on-missing-config-entry")
              // Never compress apks.
              .add("-0", "apk")
              // Add custom no-compress extensions.
              .addRepeated("-0", aaptOptions.getNoCompress())
              // Filter by resource configuration type.
              .add("-c", Joiner.on(',').join(resourceConfigs))
              // Split APKs if any splits were specified.
              .whenVersionIsAtLeast(new Revision(23))
              .thenAddRepeated("--split", splits);
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
   * directory with data binding expressions stripped out (so aapt, which doesn't understand data
   * binding, can properly read them).
   *
   * <p>Returns the resources directory that aapt should read.
   */
  static Path processDataBindings(
      Path processedResourceOutputDirectory,
      Path inputResourcesDir,
      Path dataBindingInfoOut,
      String packagePath,
      boolean shouldZipDataBindingInfo)
      throws IOException {

    if (dataBindingInfoOut == null) {
      return inputResourcesDir;
    } else if (!Files.isDirectory(inputResourcesDir)) {
      // No resources: no data binding needed. Create a dummy file to satisfy declared outputs.
      Files.createFile(dataBindingInfoOut);
      return inputResourcesDir;
    }

    // Strip the file name (the data binding library automatically adds it back in).
    // ** The data binding library assumes this file is called "layout-info.zip". **
    if (shouldZipDataBindingInfo) {
      dataBindingInfoOut = dataBindingInfoOut.getParent();
      if (Files.notExists(dataBindingInfoOut)) {
        Files.createDirectory(dataBindingInfoOut);
      }
    }

    // Create a directory for the resources, namespaced with the old resource path
    Path processedResourceDir =
        Files.createDirectories(
            processedResourceOutputDirectory.resolve(
                inputResourcesDir.isAbsolute()
                    ? inputResourcesDir.getRoot().relativize(inputResourcesDir)
                    : inputResourcesDir));

    ProcessXmlOptions options = new ProcessXmlOptions();
    options.setAppId(packagePath);
    options.setResInput(inputResourcesDir.toFile());
    options.setResOutput(processedResourceDir.toFile());
    options.setLayoutInfoOutput(dataBindingInfoOut.toFile());
    // Whether or not to aggregate data-bound .xml files into a single .zip.
    options.setZipLayoutInfo(shouldZipDataBindingInfo);

    try {
      AndroidDataBinding.doRun(options);
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
    return processedResourceDir;
  }

  public ResourceSymbols loadResourceSymbolTable(
      Iterable<? extends SymbolFileProvider> libraries,
      String appPackageName,
      Path primaryRTxt,
      Multimap<String, ResourceSymbols> libMap)
      throws IOException {
    // The reported availableProcessors may be higher than the actual resources
    // (on a shared system). On the other hand, a lot of the work is I/O, so it's not completely
    // CPU bound. As a compromise, divide by 2 the reported availableProcessors.
    int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
    ListeningExecutorService executorService =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads));
    try (Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
      for (Map.Entry<String, ListenableFuture<ResourceSymbols>> entry :
          ResourceSymbols.loadFrom(libraries, executorService, appPackageName).entries()) {
        libMap.put(entry.getKey(), entry.getValue().get());
      }
      if (primaryRTxt != null && Files.exists(primaryRTxt)) {
        return ResourceSymbols.load(primaryRTxt, executorService).get();
      }
      return ResourceSymbols.merge(libMap.values());
    } catch (InterruptedException | ExecutionException e) {
      throw new IOException("Failed to load SymbolFile: ", e);
    }
  }

  void writeDependencyPackageRJavaFiles(
      List<DependencyAndroidData> dependencyData,
      String customPackageForR,
      Path androidManifest,
      Path sourceOut)
      throws IOException {
    List<SymbolFileProvider> libraries = new ArrayList<>();
    for (DependencyAndroidData dataDep : dependencyData) {
      SymbolFileProvider library = dataDep.asSymbolFileProvider();
      libraries.add(library);
    }
    String appPackageName = customPackageForR;
    if (appPackageName == null) {
      appPackageName = VariantConfiguration.getManifestPackage(androidManifest.toFile());
    }
    Multimap<String, ResourceSymbols> libSymbolMap = ArrayListMultimap.create();
    Path primaryRTxt = sourceOut != null ? sourceOut.resolve("R.txt") : null;
    if (primaryRTxt != null && !libraries.isEmpty()) {
      ResourceSymbols fullSymbolValues =
          loadResourceSymbolTable(libraries, appPackageName, primaryRTxt, libSymbolMap);
      // Loop on all the package name, merge all the symbols to write, and write.
      for (String packageName : libSymbolMap.keySet()) {
        Collection<ResourceSymbols> symbols = libSymbolMap.get(packageName);
        fullSymbolValues.writeSourcesTo(sourceOut, packageName, symbols, /* finalFields= */ true);
      }
    }
  }

  /** Renames aapt's split outputs according to the input flags. */
  private void renameSplitPackages(Path packageOut, Iterable<String> splits)
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
    for (Map.Entry<String, String> splitMapping : outputs.entrySet()) {
      Path resultPath = packageOut.resolveSibling(prefix + splitMapping.getValue());
      if (!splitMapping.getKey().equals(splitMapping.getValue())) {
        Path sourcePath = packageOut.resolveSibling(prefix + splitMapping.getKey());
        Files.move(sourcePath, resultPath);
      }
    }
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

  /** Shutdown AOSP utilized thread-pool. */
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
