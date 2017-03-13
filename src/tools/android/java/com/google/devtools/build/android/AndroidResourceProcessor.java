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
import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.res2.MergingException;
import com.android.io.FileWrapper;
import com.android.io.StreamException;
import com.android.manifmerger.ManifestMerger2;
import com.android.manifmerger.ManifestMerger2.Invoker;
import com.android.manifmerger.ManifestMerger2.Invoker.Feature;
import com.android.manifmerger.ManifestMerger2.MergeFailureException;
import com.android.manifmerger.ManifestMerger2.MergeType;
import com.android.manifmerger.ManifestMerger2.SystemProperty;
import com.android.manifmerger.MergingReport;
import com.android.manifmerger.MergingReport.MergedManifestKind;
import com.android.manifmerger.PlaceholderHandler;
import com.android.repository.Revision;
import com.android.utils.ILogger;
import com.android.utils.Pair;
import com.android.utils.StdLogger;
import com.android.xml.AndroidManifest;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.RevisionConverter;
import com.google.devtools.build.android.ParsedAndroidData.Builder;
import com.google.devtools.build.android.SplitConfigurationFilter.UnrecognizedSplitsException;
import com.google.devtools.build.android.resources.RClassGenerator;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import javax.xml.stream.FactoryConfigurationError;
import javax.xml.stream.XMLEventFactory;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLEventWriter;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;
import javax.xml.xpath.XPathExpressionException;

/**
 * Provides a wrapper around the AOSP build tools for resource processing.
 */
public class AndroidResourceProcessor {
  private static final Logger logger = Logger.getLogger(AndroidResourceProcessor.class.getName());

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
      return ImmutableList.of();
    }
  }

  /** Shutdowns and verifies that no tasks are running in the executor service. */
  private static final class ExecutorServiceCloser implements Closeable {
    private final ListeningExecutorService executorService;
    private ExecutorServiceCloser(ListeningExecutorService executorService) {
      this.executorService = executorService;
    }

    @Override
    public void close() throws IOException {
      List<Runnable> unfinishedTasks = executorService.shutdownNow();
      if (!unfinishedTasks.isEmpty()) {
        throw new IOException(
            "Shutting down the executor with unfinished tasks:" + unfinishedTasks);
      }
    }

    public static Closeable createWith(ListeningExecutorService executorService) {
      return new ExecutorServiceCloser(executorService);
    }
  }

  private static final ImmutableMap<SystemProperty, String> SYSTEM_PROPERTY_NAMES = Maps.toMap(
      Arrays.asList(SystemProperty.values()), new Function<SystemProperty, String>() {
        @Override
        public String apply(SystemProperty property) {
          if (property == SystemProperty.PACKAGE) {
            return "applicationId";
          } else {
            return property.toCamelCase();
          }
        }
      });

  private static final Pattern HEX_REGEX = Pattern.compile("0x[0-9A-Fa-f]{8}");
  private final StdLogger stdLogger;

  public AndroidResourceProcessor(StdLogger stdLogger) {
    this.stdLogger = stdLogger;
  }

  /**
   * Copies the R.txt to the expected place.
   *
   * @param generatedSourceRoot The path to the generated R.txt.
   * @param rOutput The Path to write the R.txt.
   * @param staticIds Boolean that indicates if the ids should be set to 0x1 for caching purposes.
   */
  public void copyRToOutput(Path generatedSourceRoot, Path rOutput, boolean staticIds) {
    try {
      Files.createDirectories(rOutput.getParent());
      final Path source = generatedSourceRoot.resolve("R.txt");
      if (Files.exists(source)) {
        if (staticIds) {
          String contents =
              HEX_REGEX
                  .matcher(Joiner.on("\n").join(Files.readAllLines(source, UTF_8)))
                  .replaceAll("0x1");
          Files.write(rOutput, contents.getBytes(UTF_8));
        } else {
          Files.copy(source, rOutput);
        }
      } else {
        // The R.txt wasn't generated, create one for future inheritance, as Bazel always requires
        // outputs. This state occurs when there are no resource directories.
        Files.createFile(rOutput);
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(rOutput, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Creates a zip archive from all found R.java files.
   */
  public void createSrcJar(Path generatedSourcesRoot, Path srcJar, boolean staticIds) {
    try {
      Files.createDirectories(srcJar.getParent());
      try (final ZipOutputStream zip = new ZipOutputStream(
          new BufferedOutputStream(Files.newOutputStream(srcJar)))) {
        SymbolFileSrcJarBuildingVisitor visitor =
            new SymbolFileSrcJarBuildingVisitor(zip, generatedSourcesRoot, staticIds);
        Files.walkFileTree(generatedSourcesRoot, visitor);
        visitor.writeEntries();
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(srcJar, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Creates a zip archive from all found R.class (and inner class) files.
   */
  public void createClassJar(Path generatedClassesRoot, Path classJar) {
    try {
      Files.createDirectories(classJar.getParent());
      try (final ZipOutputStream zip = new ZipOutputStream(
          new BufferedOutputStream(Files.newOutputStream(classJar)))) {
        ClassJarBuildingVisitor visitor = new ClassJarBuildingVisitor(zip, generatedClassesRoot);
        Files.walkFileTree(generatedClassesRoot, visitor);
        visitor.writeEntries();
        visitor.writeManifestContent();
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(classJar, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Copies the AndroidManifest.xml to the specified output location.
   *
   * @param androidData The MergedAndroidData which contains the manifest to be written to
   *    manifestOut.
   * @param manifestOut The Path to write the AndroidManifest.xml.
   */
  public void copyManifestToOutput(MergedAndroidData androidData, Path manifestOut) {
    try {
      Files.createDirectories(manifestOut.getParent());
      Files.copy(androidData.getManifest(), manifestOut);
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(manifestOut, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Creates a zip file containing the provided android resources and assets.
   *
   * @param resourcesRoot The root containing android resources to be written.
   * @param assetsRoot The root containing android assets to be written.
   * @param output The path to write the zip file
   * @param compress Whether or not to compress the content
   * @throws IOException
   */
  public void createResourcesZip(Path resourcesRoot, Path assetsRoot, Path output, boolean compress)
      throws IOException {
    try (ZipOutputStream zout = new ZipOutputStream(
        new BufferedOutputStream(Files.newOutputStream(output)))) {
      if (Files.exists(resourcesRoot)) {
        ZipBuilderVisitor visitor = new ZipBuilderVisitor(zout, resourcesRoot, "res");
        visitor.setCompress(compress);
        Files.walkFileTree(resourcesRoot, visitor);
        visitor.writeEntries();
      }
      if (Files.exists(assetsRoot)) {
        ZipBuilderVisitor visitor = new ZipBuilderVisitor(zout, assetsRoot, "assets");
        visitor.setCompress(compress);
        Files.walkFileTree(assetsRoot, visitor);
        visitor.writeEntries();
      }
    }
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

  public MergedAndroidData processManifest(
      VariantType variantType,
      String customPackageForR,
      String applicationId,
      int versionCode,
      String versionName,
      MergedAndroidData primaryData,
      Path processedManifest)
      throws IOException {

    ManifestMerger2.MergeType mergeType =
        variantType == VariantType.DEFAULT
            ? ManifestMerger2.MergeType.APPLICATION
            : ManifestMerger2.MergeType.LIBRARY;

    String newManifestPackage =
        variantType == VariantType.DEFAULT ? applicationId : customPackageForR;

    if (versionCode != -1 || versionName != null || newManifestPackage != null) {
      Files.createDirectories(processedManifest.getParent());

      // The generics on Invoker don't make sense, so ignore them.
      @SuppressWarnings("unchecked")
      Invoker<?> manifestMergerInvoker =
          ManifestMerger2.newMerger(primaryData.getManifest().toFile(), stdLogger, mergeType);
      // Stamp new package
      if (newManifestPackage != null) {
        manifestMergerInvoker.setOverride(SystemProperty.PACKAGE, newManifestPackage);
      }
      // Stamp version and applicationId (if provided) into the manifest
      if (versionCode > 0) {
        manifestMergerInvoker.setOverride(SystemProperty.VERSION_CODE, String.valueOf(versionCode));
      }
      if (versionName != null) {
        manifestMergerInvoker.setOverride(SystemProperty.VERSION_NAME, versionName);
      }

      MergedManifestKind mergedManifestKind = MergedManifestKind.MERGED;
      if (mergeType == ManifestMerger2.MergeType.APPLICATION) {
        manifestMergerInvoker.withFeatures(Invoker.Feature.REMOVE_TOOLS_DECLARATIONS);
      }

      try {
        MergingReport mergingReport = manifestMergerInvoker.merge();
        switch (mergingReport.getResult()) {
          case WARNING:
            mergingReport.log(stdLogger);
            writeMergedManifest(mergedManifestKind, mergingReport, processedManifest);
            break;
          case SUCCESS:
            writeMergedManifest(mergedManifestKind, mergingReport, processedManifest);
            break;
          case ERROR:
            mergingReport.log(stdLogger);
            throw new RuntimeException(mergingReport.getReportString());
          default:
            throw new RuntimeException("Unhandled result type : " + mergingReport.getResult());
        }
      } catch (IOException | MergeFailureException e) {
        throw new RuntimeException(e);
      }
      return new MergedAndroidData(
          primaryData.getResourceDir(), primaryData.getAssetDir(), processedManifest);
    }
    return primaryData;
  }

  /**
   * A logger that will print messages to a target OutputStream.
   */
  private static final class PrintStreamLogger implements ILogger {
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

  /**
   * Merge several manifests into one and perform placeholder substitutions. This operation uses
   * Gradle semantics.
   *
   * @param manifest The primary manifest of the merge.
   * @param mergeeManifests Manifests to be merged into {@code manifest}.
   * @param mergeType Whether the merger should operate in application or library mode.
   * @param values A map of strings to be used as manifest placeholders and overrides. packageName
   *     is the only disallowed value and will be ignored.
   * @param output The path to write the resultant manifest to.
   * @param logFile The path to write the merger log to.
   * @return The path of the resultant manifest, either {@code output}, or {@code manifest} if no
   *     merging was required.
   * @throws IOException if there was a problem writing the merged manifest.
   */
  public Path mergeManifest(
      Path manifest,
      Map<Path, String> mergeeManifests,
      MergeType mergeType,
      Map<String, String> values,
      Path output,
      Path logFile)
      throws IOException {
    if (mergeeManifests.isEmpty() && values.isEmpty()) {
      return manifest;
    }

    Invoker<?> manifestMerger = ManifestMerger2.newMerger(manifest.toFile(), stdLogger, mergeType);
    MergedManifestKind mergedManifestKind = MergedManifestKind.MERGED;
    if (mergeType == MergeType.APPLICATION) {
      manifestMerger.withFeatures(Feature.REMOVE_TOOLS_DECLARATIONS);
    }

    // Add mergee manifests
    List<Pair<String, File>> libraryManifests = new ArrayList<>();
    for (Entry<Path, String> mergeeManifest : mergeeManifests.entrySet()) {
      libraryManifests.add(Pair.of(mergeeManifest.getValue(), mergeeManifest.getKey().toFile()));
    }
    manifestMerger.addLibraryManifests(libraryManifests);

    // Extract SystemProperties from the provided values.
    Map<String, Object> placeholders = new HashMap<>();
    placeholders.putAll(values);
    for (SystemProperty property : SystemProperty.values()) {
      if (values.containsKey(SYSTEM_PROPERTY_NAMES.get(property))) {
        manifestMerger.setOverride(property, values.get(SYSTEM_PROPERTY_NAMES.get(property)));

        // The manifest merger does not allow explicitly specifying either applicationId or
        // packageName as placeholders if SystemProperty.PACKAGE is specified. It forces these
        // placeholders to have the same value as specified by SystemProperty.PACKAGE.
        if (property == SystemProperty.PACKAGE) {
          placeholders.remove(PlaceholderHandler.APPLICATION_ID);
          placeholders.remove(PlaceholderHandler.PACKAGE_NAME);
        }
      }
    }

    // Add placeholders for all values.
    // packageName is populated from either the applicationId override or from the manifest itself;
    // it cannot be manually specified.
    placeholders.remove(PlaceholderHandler.PACKAGE_NAME);
    manifestMerger.setPlaceHolderValues(placeholders);

    try {
      MergingReport mergingReport = manifestMerger.merge();

      if (logFile != null) {
        logFile.getParent().toFile().mkdirs();
        try (PrintStream stream = new PrintStream(logFile.toFile())) {
          mergingReport.log(new PrintStreamLogger(stream));
        }
      }
      switch (mergingReport.getResult()) {
        case WARNING:
          mergingReport.log(stdLogger);
          Files.createDirectories(output.getParent());
          writeMergedManifest(mergedManifestKind, mergingReport, output);
          break;
        case SUCCESS:
          Files.createDirectories(output.getParent());
          writeMergedManifest(mergedManifestKind, mergingReport, output);
          break;
        case ERROR:
          mergingReport.log(stdLogger);
          throw new RuntimeException(mergingReport.getReportString());
        default:
          throw new RuntimeException("Unhandled result type : " + mergingReport.getResult());
      }
    } catch (MergeFailureException e) {
      throw new RuntimeException(e);
    }

    return output;
  }

  private void writeMergedManifest(
      MergedManifestKind mergedManifestKind, MergingReport mergingReport, Path manifestOut)
      throws IOException {
    String manifestContents = mergingReport.getMergedDocument(mergedManifestKind);
    String annotatedDocument = mergingReport.getMergedDocument(MergedManifestKind.BLAME);
    stdLogger.verbose(annotatedDocument);
    Files.write(manifestOut, manifestContents.getBytes(UTF_8));
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
   * Overwrite the package attribute of {@code <manifest>} in an AndroidManifest.xml file.
   *
   * @param manifest The input manifest.
   * @param customPackage The package to write to the manifest.
   * @param output The output manifest to generate.
   * @return The output manifest if generated or the input manifest if no overwriting is required.
   */
  /* TODO(apell): switch from custom xml parsing to Gradle merger with NO_PLACEHOLDER_REPLACEMENT
   * set when android common is updated to version 2.5.0.
   */
  public Path writeManifestPackage(Path manifest, String customPackage, Path output) {
    if (Strings.isNullOrEmpty(customPackage)) {
      return manifest;
    }
    try {
      Files.createDirectories(output.getParent());
      XMLEventReader reader =
          XMLInputFactory.newInstance()
              .createXMLEventReader(Files.newInputStream(manifest), UTF_8.name());
      XMLEventWriter writer =
          XMLOutputFactory.newInstance()
              .createXMLEventWriter(Files.newOutputStream(output), UTF_8.name());
      XMLEventFactory eventFactory = XMLEventFactory.newInstance();
      while (reader.hasNext()) {
        XMLEvent event = reader.nextEvent();
        if (event.isStartElement()
            && event.asStartElement().getName().toString().equalsIgnoreCase("manifest")) {
          StartElement element = event.asStartElement();
          @SuppressWarnings("unchecked")
          Iterator<Attribute> attributes = element.getAttributes();
          ImmutableList.Builder<Attribute> newAttributes = ImmutableList.builder();
          while (attributes.hasNext()) {
            Attribute attr = attributes.next();
            if (attr.getName().toString().equalsIgnoreCase("package")) {
              newAttributes.add(eventFactory.createAttribute("package", customPackage));
            } else {
              newAttributes.add(attr);
            }
          }
          writer.add(
              eventFactory.createStartElement(
                  element.getName(), newAttributes.build().iterator(), element.getNamespaces()));
        } else {
          writer.add(event);
        }
      }
      writer.flush();
    } catch (XMLStreamException | FactoryConfigurationError | IOException e) {
      throw new RuntimeException(e);
    }

    return output;
  }

  /**
   * Merges all secondary resources with the primary resources, given that the primary resources
   * have not yet been parsed and serialized.
   */
  public MergedAndroidData mergeData(
      final UnvalidatedAndroidData primary,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut)
      throws MergingException {
    try {
      final ParsedAndroidData parsedPrimary = ParsedAndroidData.from(primary);
      return mergeData(parsedPrimary, primary.getManifest(), direct, transitive,
          resourcesOut, assetsOut, cruncher, type, symbolsOut, null /* rclassWriter */);
    } catch (IOException e) {
      throw MergingException.wrapException(e).build();
    }
  }

  /**
   * Merges all secondary resources with the primary resources, given that the primary resources
   * have been separately parsed and serialized.
   */
  public MergedAndroidData mergeData(
      final SerializedAndroidData primary,
      final Path primaryManifest,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      @Nullable final AndroidResourceClassWriter rclassWriter)
      throws MergingException {
    final ParsedAndroidData.Builder primaryBuilder = ParsedAndroidData.Builder.newBuilder();
    final AndroidDataDeserializer deserializer = AndroidDataDeserializer.create();
    primary.deserialize(deserializer, primaryBuilder.consumers());
    ParsedAndroidData primaryData = primaryBuilder.build();
    return mergeData(
        primaryData,
        primaryManifest,
        direct,
        transitive,
        resourcesOut,
        assetsOut,
        cruncher,
        type,
        symbolsOut,
        rclassWriter);
  }

  /**
   * Merges all secondary resources with the primary resources.
   */
  private MergedAndroidData mergeData(
      final ParsedAndroidData primary,
      final Path primaryManifest,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      @Nullable AndroidResourceClassWriter rclassWriter)
      throws MergingException {
    Stopwatch timer = Stopwatch.createStarted();
    final ListeningExecutorService executorService =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(15));
    try (Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
      AndroidDataMerger merger = AndroidDataMerger.createWithPathDeduplictor(executorService);
      UnwrittenMergedAndroidData merged =
          merger.loadAndMerge(
              transitive,
              direct,
              primary,
              primaryManifest,
              type != VariantType.LIBRARY);
      logger.fine(String.format("merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      timer.reset().start();
      if (symbolsOut != null) {
        AndroidDataSerializer serializer = AndroidDataSerializer.create();
        merged.serializeTo(serializer);
        serializer.flushTo(symbolsOut);
        logger.fine(
            String.format(
                "serialize merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        timer.reset().start();
      }
      if (rclassWriter != null) {
        merged.writeResourceClass(rclassWriter);
        logger.fine(
            String.format("write classes finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        timer.reset().start();
      }
      AndroidDataWriter writer =
          AndroidDataWriter.createWith(
              resourcesOut.getParent(), resourcesOut, assetsOut, cruncher, executorService);
      return merged.write(writer);
    } catch (IOException e) {
      throw MergingException.wrapException(e).build();
    } finally {
      logger.fine(
          String.format("write merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
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

  /** Deserializes a list of serialized resource paths to a {@link ParsedAndroidData}. */
  public ParsedAndroidData deserializeSymbolsToData(List<Path> symbolPaths)
      throws IOException, MergingException {
    AndroidDataDeserializer deserializer = AndroidDataDeserializer.create();
    final ListeningExecutorService executorService =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(15));
    final Builder deserializedDataBuilder = ParsedAndroidData.Builder.newBuilder();
    try (Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
      List<ListenableFuture<Boolean>> deserializing = new ArrayList<>();
      for (final Path symbolPath : symbolPaths) {
        deserializing.add(
            executorService.submit(
                new Deserialize(deserializer, symbolPath, deserializedDataBuilder)));
      }
      FailedFutureAggregator<MergingException> aggregator =
          FailedFutureAggregator.createForMergingExceptionWithMessage(
              "Failure(s) during dependency parsing");
      aggregator.aggregateAndMaybeThrow(deserializing);
    }
    return deserializedDataBuilder.build();
  }

  /**
   * A FileVisitor that will add all files to be stored in a zip archive.
   */
  private static class ZipBuilderVisitor extends SimpleFileVisitor<Path> {

    // The earliest date representable in a zip file, 1-1-1980 (the DOS epoch).
    private static final long ZIP_EPOCH = 315561600000L;
    // ZIP timestamps have a resolution of 2 seconds.
    // see http://www.info-zip.org/FAQ.html#limits
    private static final long MINIMUM_TIMESTAMP_INCREMENT = 2000L;

    private final ZipOutputStream zip;
    protected final Path root;
    private final String directoryPrefix;
    private int storageMethod = ZipEntry.STORED;
    private final Collection<Path> paths = new ArrayList<>();

    ZipBuilderVisitor(ZipOutputStream zip, Path root, String directory) {
      this.zip = zip;
      this.root = root;
      this.directoryPrefix = directory;
    }

    public void setCompress(boolean compress) {
      storageMethod = compress ? ZipEntry.DEFLATED : ZipEntry.STORED;
    }

    /**
     * Iterate through collected file paths in a deterministic order and write to the zip.
     *
     * @throws IOException if there is an error reading from the source or writing to the zip.
     */
    void writeEntries() throws IOException {
      for (Path path : Ordering.natural().immutableSortedCopy(paths)) {
        writeFileEntry(path);
      }
    }

    /**
     * Normalize timestamps for deterministic builds. Stamp .class files to be a bit newer
     * than .java files. See:
     * {@link com.google.devtools.build.buildjar.jarhelper.JarHelper#normalizedTimestamp(String)}
     */
    protected long normalizeTime(String filename) {
      if (filename.endsWith(".class")) {
        return ZIP_EPOCH + MINIMUM_TIMESTAMP_INCREMENT;
      } else {
        return ZIP_EPOCH;
      }
    }

    protected void addEntry(Path file, byte[] content) throws IOException {
      String prefix = directoryPrefix != null ? (directoryPrefix + "/") : "";
      String relativeName = root.relativize(file).toString();
      ZipEntry entry = new ZipEntry(prefix + relativeName);
      entry.setMethod(storageMethod);
      entry.setTime(normalizeTime(relativeName));
      entry.setSize(content.length);
      CRC32 crc32 = new CRC32();
      crc32.update(content);
      entry.setCrc(crc32.getValue());

      zip.putNextEntry(entry);
      zip.write(content);
      zip.closeEntry();
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      paths.add(file);
      return FileVisitResult.CONTINUE;
    }

    protected void writeFileEntry(Path file) throws IOException {
      byte[] content = Files.readAllBytes(file);
      addEntry(file, content);
    }
  }

  /**
   * A FileVisitor that will add all R.java files to be stored in a zip archive.
   */
  private static final class SymbolFileSrcJarBuildingVisitor extends ZipBuilderVisitor {

    static final Pattern PACKAGE_PATTERN =
        Pattern.compile("\\s*package ([a-zA-Z_$][a-zA-Z\\d_$]*(?:\\.[a-zA-Z_$][a-zA-Z\\d_$]*)*)");
    static final Pattern ID_PATTERN =
        Pattern.compile("public static int ([\\w\\.]+)=0x[0-9A-fa-f]+;");
    static final Pattern INNER_CLASS =
        Pattern.compile("public static class ([a-z_]*) \\{(.*?)\\}", Pattern.DOTALL);

    private final boolean staticIds;

    private SymbolFileSrcJarBuildingVisitor(ZipOutputStream zip, Path root, boolean staticIds) {
      super(zip, root, null);
      this.staticIds = staticIds;
    }

    private String replaceIdsWithStaticIds(String contents) {
      Matcher packageMatcher = PACKAGE_PATTERN.matcher(contents);
      if (!packageMatcher.find()) {
        return contents;
      }
      String pkg = packageMatcher.group(1);
      StringBuffer out = new StringBuffer();
      Matcher innerClassMatcher = INNER_CLASS.matcher(contents);
      while (innerClassMatcher.find()) {
        String resourceType = innerClassMatcher.group(1);
        Matcher idMatcher = ID_PATTERN.matcher(innerClassMatcher.group(2));
        StringBuffer resourceIds = new StringBuffer();
        while (idMatcher.find()) {
          String javaId = idMatcher.group(1);
          idMatcher.appendReplacement(
              resourceIds,
              String.format(
                  "public static int %s=0x%08X;", javaId, Objects.hash(pkg, resourceType, javaId)));
        }
        idMatcher.appendTail(resourceIds);
        innerClassMatcher.appendReplacement(
            out,
            String.format("public static class %s {%s}", resourceType, resourceIds.toString()));
      }
      innerClassMatcher.appendTail(out);
      return out.toString();
    }

    @Override
    protected void writeFileEntry(Path file) throws IOException {
      if (file.getFileName().endsWith("R.java")) {
        byte[] content = Files.readAllBytes(file);
        if (staticIds) {
          content =
              replaceIdsWithStaticIds(UTF_8.decode(ByteBuffer.wrap(content)).toString())
                  .getBytes(UTF_8);
        }
        addEntry(file, content);
      }
    }
  }

  /**
   * A FileVisitor that will add all R class files to be stored in a zip archive.
   */
  private static final class ClassJarBuildingVisitor extends ZipBuilderVisitor {

    ClassJarBuildingVisitor(ZipOutputStream zip, Path root) {
      super(zip, root, null);
    }

    @Override
    protected void writeFileEntry(Path file) throws IOException {
      Path filename = file.getFileName();
      String name = filename.toString();
      if (name.endsWith(".class")) {
        byte[] content = Files.readAllBytes(file);
        addEntry(file, content);
      }
    }

    private byte[] manifestContent() throws IOException {
      Manifest manifest = new Manifest();
      Attributes attributes = manifest.getMainAttributes();
      attributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");
      Attributes.Name createdBy = new Attributes.Name("Created-By");
      if (attributes.getValue(createdBy) == null) {
        attributes.put(createdBy, "bazel");
      }
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      manifest.write(out);
      return out.toByteArray();
    }

    void writeManifestContent() throws IOException {
      addEntry(root.resolve(JarFile.MANIFEST_NAME), manifestContent());
    }
  }

  /** Task to deserialize resources from a path. */
  private static final class Deserialize implements Callable<Boolean> {

    private final Path symbolPath;

    private final Builder finalDataBuilder;
    private final AndroidDataDeserializer deserializer;

    private Deserialize(
        AndroidDataDeserializer deserializer, Path symbolPath, Builder finalDataBuilder) {
      this.deserializer = deserializer;
      this.symbolPath = symbolPath;
      this.finalDataBuilder = finalDataBuilder;
    }

    @Override
    public Boolean call() throws Exception {
      final Builder parsedDataBuilder = ParsedAndroidData.Builder.newBuilder();
      deserializer.read(symbolPath, parsedDataBuilder.consumers());
      // The builder isn't threadsafe, so synchronize the copyTo call.
      synchronized (finalDataBuilder) {
        // All the resources are sorted before writing, so they can be aggregated in
        // whatever order here.
        parsedDataBuilder.copyTo(finalDataBuilder);
      }
      return Boolean.TRUE;
    }
  }
}
