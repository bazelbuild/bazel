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

import com.google.common.base.Joiner;
import com.google.common.base.Throwables;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.FullRevisionConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;

import com.android.annotations.Nullable;
import com.android.builder.core.VariantConfiguration;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.builder.internal.SymbolLoader;
import com.android.builder.internal.SymbolWriter;
import com.android.builder.model.AaptOptions;
import com.android.ide.common.internal.CommandLineRunner;
import com.android.ide.common.internal.ExecutorSingleton;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.res2.AssetMerger;
import com.android.ide.common.res2.AssetSet;
import com.android.ide.common.res2.MergedAssetWriter;
import com.android.ide.common.res2.MergedResourceWriter;
import com.android.ide.common.res2.MergingException;
import com.android.ide.common.res2.ResourceMerger;
import com.android.ide.common.res2.ResourceSet;
import com.android.manifmerger.ManifestMerger2;
import com.android.manifmerger.ManifestMerger2.Invoker;
import com.android.manifmerger.ManifestMerger2.MergeFailureException;
import com.android.manifmerger.ManifestMerger2.SystemProperty;
import com.android.manifmerger.MergingReport;
import com.android.manifmerger.XmlDocument;
import com.android.sdklib.repository.FullRevision;
import com.android.utils.StdLogger;

import org.xml.sax.SAXException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import javax.xml.parsers.ParserConfigurationException;

/**
 * Provides a wrapper around the AOSP build tools for resource processing.
 */
public class AndroidResourceProcessor {
  /**
   * Options class containing flags for Aapt setup.
   */
  public static final class AaptConfigOptions extends OptionsBase {
    @Option(name = "buildToolsVersion",
        defaultValue = "null",
        converter = FullRevisionConverter.class,
        category = "config",
        help = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2")
    public FullRevision buildToolsVersion;

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
    public boolean getUseAaptPngCruncher() {
      return options.useAaptCruncher != TriState.NO;
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
  }

  private static final Pattern HEX_REGEX = Pattern.compile("0x[0-9A-Fa-f]{8}");
  private final StdLogger stdLogger;

  public AndroidResourceProcessor(StdLogger stdLogger) {
    this.stdLogger = stdLogger;
  }

  /**
   * Copies the R.txt to the expected place.
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
          String contents = HEX_REGEX.matcher(Joiner.on("\n").join(
              Files.readAllLines(source, StandardCharsets.UTF_8))).replaceAll("0x1");
          Files.write(rOutput, contents.getBytes(StandardCharsets.UTF_8));
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
      Throwables.propagate(e);
    }
  }

  /**
   * Creates a zip archive from all found R.java files.
   */
  public void createSrcJar(Path generatedSourcesRoot, Path srcJar, boolean staticIds) {
    try {
      Files.createDirectories(srcJar.getParent());
      try (final ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(srcJar))) {
        Files.walkFileTree(generatedSourcesRoot,
            new SymbolFileSrcJarBuildingVisitor(zip, generatedSourcesRoot, staticIds));
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(srcJar, FileTime.fromMillis(0L));
    } catch (IOException e) {
      Throwables.propagate(e);
    }
  }

  /**
   * Copies the AndroidManifest.xml to the specified output location.
   *
   * @param androidData The MergedAndroidData which contains the manifest to be written to
   *     manifestOut.
   * @param manifestOut The Path to write the AndroidManifest.xml.
   */
  public void copyManifestToOutput(MergedAndroidData androidData, Path manifestOut) {
    try {
      Files.createDirectories(manifestOut.getParent());
      Files.copy(androidData.getManifest(), manifestOut);
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(manifestOut, FileTime.fromMillis(0L));
    } catch (IOException e) {
      Throwables.propagate(e);
    }
  }

  /**
   * Creates a zip file containing the provided android resources and assets.
   *
   * @param resourcesRoot The root containing android resources to be written.
   * @param assetsRoot The root containing android assets to be written.
   * @param output The path to write the zip file
   * @throws IOException
   */
  public void createResourcesZip(Path resourcesRoot, Path assetsRoot, Path output)
      throws IOException {
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(output.toFile()))) {
      if (Files.exists(resourcesRoot)) {
        Files.walkFileTree(resourcesRoot, new ZipBuilderVisitor(zout, resourcesRoot, "res"));
      }
      if (Files.exists(assetsRoot)) {
        Files.walkFileTree(assetsRoot, new ZipBuilderVisitor(zout, assetsRoot, "assets"));
      }
    }
  }

  // TODO(bazel-team): Clean up this method call -- 13 params is too many.
  /**
   * Processes resources for generated sources, configs and packaging resources.
   */
  public void processResources(
      Path aapt,
      Path androidJar,
      @Nullable FullRevision buildToolsVersion,
      VariantConfiguration.Type variantType,
      boolean debug,
      String customPackageForR,
      AaptOptions aaptOptions,
      Collection<String> resourceConfigs,
      MergedAndroidData primaryData,
      List<DependencyAndroidData> dependencyData,
      Path sourceOut,
      Path packageOut,
      Path proguardOut,
      Path publicResourcesOut)
      throws IOException, InterruptedException, LoggedErrorException {
    List<SymbolFileProvider> libraries = new ArrayList<>();
    List<String> packages = new ArrayList<>();
    for (DependencyAndroidData dataDep : dependencyData) {
      SymbolFileProvider library = dataDep.asSymbolFileProvider();
      libraries.add(library);
      packages.add(VariantConfiguration.getManifestPackage(library.getManifest()));
    }

    Path androidManifest = primaryData.getManifest();
    Path resourceDir = primaryData.getResourceDir();
    Path assetsDir = primaryData.getAssetDir();
    if (publicResourcesOut != null) {
      prepareOutputPath(publicResourcesOut.getParent());
    }

    AaptCommandBuilder commandBuilder =
        new AaptCommandBuilder(aapt, buildToolsVersion, variantType, "package")
            // If the logger is verbose, set aapt to be verbose
        .maybeAdd("-v", stdLogger.getLevel() == StdLogger.Level.VERBOSE)
        // Overwrite existing files, if they exist.
        .add("-f")
        // Resources are precrunched in the merge process.
        .add("--no-crunch")
        // Do not automatically generate versioned copies of vector XML resources.
        .maybeAdd("--no-version-vectors", new FullRevision(23))
        // Add the android.jar as a base input.
        .add("-I", androidJar)
        // Add the manifest for validation.
        .add("-M", androidManifest.toAbsolutePath())
        // Maybe add the resources if they exist
        .maybeAdd("-S", resourceDir, Files.isDirectory(resourceDir))
        // Maybe add the assets if they exist
        .maybeAdd("-A", assetsDir, Files.isDirectory(assetsDir))
        // Outputs
        .maybeAdd("-m", sourceOut != null)
        .maybeAdd("-J", prepareOutputPath(sourceOut), sourceOut != null)
        .maybeAdd("--output-text-symbols", prepareOutputPath(sourceOut), sourceOut != null)
        .add("-F", packageOut)
        .add("-G", proguardOut)
        .add("-P", publicResourcesOut)
        .maybeAdd("--debug-mode", debug)
        .add("--custom-package", customPackageForR)
        // If it is a library, do not generate final java ids.
        .maybeAdd("--non-constant-id", VariantConfiguration.Type.LIBRARY)
        // Generate the dependent R and Manifest files.
        .maybeAdd("--extra-packages", Joiner.on(":").join(packages),
            VariantConfiguration.Type.DEFAULT)
        .add("--ignore-assets", aaptOptions.getIgnoreAssets())
        .maybeAdd("--error-on-missing-config-entry", aaptOptions.getFailOnMissingConfigEntry())
        // Never compress apks.
        .add("-0", "apk")
        // Add custom no-compress extensions.
        .addRepeated("-0", aaptOptions.getNoCompress())
        // Filter by resource configuration type.
        .add("-c", Joiner.on(',').join(resourceConfigs));

    new CommandLineRunner(stdLogger).runCmdLine(commandBuilder.build(), null);

    // The R needs to be created for each library in the dependencies,
    // but only if the current project is not a library.
    writeDependencyPackageRs(variantType, customPackageForR, libraries, androidManifest.toFile(),
        sourceOut);

    // Reset the output date stamps.
    if (proguardOut != null) {
      Files.setLastModifiedTime(proguardOut, FileTime.fromMillis(0L));
    }
    if (packageOut != null) {
      Files.setLastModifiedTime(packageOut, FileTime.fromMillis(0L));
    }
    if (publicResourcesOut != null && Files.exists(publicResourcesOut)) {
      Files.setLastModifiedTime(publicResourcesOut, FileTime.fromMillis(0L));
    }
  }

  private void writeDependencyPackageRs(VariantConfiguration.Type variantType,
      String customPackageForR, List<SymbolFileProvider> libraries, File androidManifest,
      Path sourceOut) throws IOException {
    if (sourceOut != null && variantType != VariantConfiguration.Type.LIBRARY
        && !libraries.isEmpty()) {
      SymbolLoader fullSymbolValues = null;

      String appPackageName = customPackageForR;
      if (appPackageName == null) {
        appPackageName = VariantConfiguration.getManifestPackage(androidManifest);
      }

      // List of all the symbol loaders per package names.
      Multimap<String, SymbolLoader> libMap = ArrayListMultimap.create();

      for (SymbolFileProvider lib : libraries) {
        String packageName = VariantConfiguration.getManifestPackage(lib.getManifest());

        // If the library package matches the app package skip -- the R class will contain
        // all the possible resources so it will not need to generate a new R.
        if (appPackageName.equals(packageName)) {
          continue;
        }

        File rFile = lib.getSymbolFile();
        // If the library has no resource, this file won't exist.
        if (rFile.isFile()) {
          // Load the full values if that's not already been done.
          // Doing it lazily allow us to support the case where there's no
          // resources anywhere.
          if (fullSymbolValues == null) {
            fullSymbolValues = new SymbolLoader(sourceOut.resolve("R.txt").toFile(), stdLogger);
            fullSymbolValues.load();
          }

          SymbolLoader libSymbols = new SymbolLoader(rFile, stdLogger);
          libSymbols.load();

          // store these symbols by associating them with the package name.
          libMap.put(packageName, libSymbols);
        }
      }

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
  }

  public MergedAndroidData processManifest(
      VariantConfiguration.Type variantType,
      String customPackageForR,
      String applicationId,
      int versionCode,
      String versionName,
      MergedAndroidData primaryData,
      Path processedManifest) throws IOException {

    ManifestMerger2.MergeType mergeType = variantType == VariantConfiguration.Type.DEFAULT
        ? ManifestMerger2.MergeType.APPLICATION : ManifestMerger2.MergeType.LIBRARY;

    String newManifestPackage = variantType == VariantConfiguration.Type.DEFAULT
        ? applicationId : customPackageForR;

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

      if (mergeType == ManifestMerger2.MergeType.APPLICATION) {
        manifestMergerInvoker.withFeatures(Invoker.Feature.REMOVE_TOOLS_DECLARATIONS);
      }

      try {
        MergingReport mergingReport = manifestMergerInvoker.merge();
        switch (mergingReport.getResult()) {
          case WARNING:
            mergingReport.log(stdLogger);
            writeMergedManifest(mergingReport, processedManifest);
            break;
          case SUCCESS:
            writeMergedManifest(mergingReport, processedManifest);
            break;
          case ERROR:
            mergingReport.log(stdLogger);
            throw new RuntimeException(mergingReport.getReportString());
          default:
            throw new RuntimeException("Unhandled result type : " + mergingReport.getResult());
        }
      } catch (
          IOException | SAXException | ParserConfigurationException | MergeFailureException e) {
        Throwables.propagate(e);
      }
      return new MergedAndroidData(primaryData.getResourceDir(), primaryData.getAssetDir(),
          processedManifest);
    }
    return primaryData;
  }

  private void writeMergedManifest(MergingReport mergingReport,
      Path manifestOut) throws IOException, SAXException, ParserConfigurationException {
    XmlDocument xmlDocument = mergingReport.getMergedDocument().get();
    String annotatedDocument = mergingReport.getActions().blame(xmlDocument);
    stdLogger.verbose(annotatedDocument);
    Files.write(
        manifestOut, xmlDocument.prettyPrint().getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Merges all secondary resources with the primary resources.
   */
  public MergedAndroidData mergeData(
      final UnvalidatedAndroidData primary,
      final List<DependencyAndroidData> secondary,
      final Path resourcesOut,
      final Path assetsOut,
      final ImmutableList<DirectoryModifier> modifiers,
      @Nullable final PngCruncher cruncher,
      final boolean strict) throws MergingException {

    List<ResourceSet> resourceSets = new ArrayList<>();
    List<AssetSet> assetSets = new ArrayList<>();

    if (strict) {
      androidDataToStrictMergeSet(primary, secondary, modifiers, resourceSets, assetSets);
    } else {
      androidDataToRelaxedMergeSet(primary, secondary, modifiers, resourceSets, assetSets);
    }
    ResourceMerger merger = new ResourceMerger();
    for (ResourceSet set : resourceSets) {
      set.loadFromFiles(stdLogger);
      merger.addDataSet(set);
    }

    AssetMerger assetMerger = new AssetMerger();
    for (AssetSet set : assetSets) {
      set.loadFromFiles(stdLogger);
      assetMerger.addDataSet(set);
    }

    MergedResourceWriter resourceWriter = new MergedResourceWriter(resourcesOut.toFile(), cruncher);
    MergedAssetWriter assetWriter = new MergedAssetWriter(assetsOut.toFile());

    merger.mergeData(resourceWriter, false);
    assetMerger.mergeData(assetWriter, false);

    return new MergedAndroidData(resourcesOut, assetsOut, primary.getManifest());
  }

  /**
   * Shutdown AOSP utilized thread-pool.
   */
  public void shutdown() {
    // AOSP code never shuts down its singleton executor and leaves the process hanging.
    ExecutorSingleton.getExecutor().shutdownNow();
  }

  private void androidDataToRelaxedMergeSet(UnvalidatedAndroidData primary,
      List<DependencyAndroidData> secondary, ImmutableList<DirectoryModifier> modifiers,
      List<ResourceSet> resourceSets, List<AssetSet> assetSets) {

    for (DependencyAndroidData dependency : secondary) {
      DependencyAndroidData modifiedDependency = dependency.modify(modifiers);
      modifiedDependency.addAsResourceSets(resourceSets);
      modifiedDependency.addAsAssetSets(assetSets);
    }
    UnvalidatedAndroidData modifiedPrimary = primary.modify(modifiers);
    modifiedPrimary.addAsResourceSets(resourceSets);
    modifiedPrimary.addAsAssetSets(assetSets);

  }

  private void androidDataToStrictMergeSet(UnvalidatedAndroidData primary,
      List<DependencyAndroidData> secondary, ImmutableList<DirectoryModifier> modifiers,
      List<ResourceSet> resourceSets, List<AssetSet> assetSets) {
    UnvalidatedAndroidData modifiedPrimary = primary.modify(modifiers);
    ResourceSet mainResources = modifiedPrimary.addToResourceSet(new ResourceSet("main"));
    AssetSet mainAssets = modifiedPrimary.addToAssets(new AssetSet("main"));
    ResourceSet dependentResources = new ResourceSet("deps");
    AssetSet dependentAssets = new AssetSet("deps");
    for (DependencyAndroidData dependency : secondary) {
      DependencyAndroidData modifiedDependency = dependency.modify(modifiers);
      modifiedDependency.addToResourceSet(dependentResources);
      modifiedDependency.addToAssets(dependentAssets);
    }
    resourceSets.add(dependentResources);
    resourceSets.add(mainResources);
    assetSets.add(dependentAssets);
    assetSets.add(mainAssets);
  }

  @Nullable private Path prepareOutputPath(@Nullable Path out) throws IOException {
    if (out == null) {
      return null;
    }
    return Files.createDirectories(out);
  }

  /**
   * A FileVisitor that will add all R.java files to be stored in a zip archive.
   */
  private static final class SymbolFileSrcJarBuildingVisitor extends SimpleFileVisitor<Path> {
    static final Pattern PACKAGE_PATTERN = Pattern.compile(
        "\\s*package ([a-zA-Z_$][a-zA-Z\\d_$]*(?:\\.[a-zA-Z_$][a-zA-Z\\d_$]*)*)");
    static final Pattern ID_PATTERN = Pattern.compile(
        "public static int ([\\w\\.]+)=0x[0-9A-fa-f]+;");
    static final Pattern INNER_CLASS = Pattern.compile("public static class ([a-z_]*) \\{(.*?)\\}",
        Pattern.DOTALL);

    // The earliest date representable in a zip file, 1-1-1980.
    private static final long ZIP_EPOCH = 315561600000L;
    private final ZipOutputStream zip;
    private final Path root;
    private final boolean staticIds;

    private SymbolFileSrcJarBuildingVisitor(ZipOutputStream zip, Path root, boolean staticIds) {
      this.zip = zip;
      this.root = root;
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
          idMatcher.appendReplacement(resourceIds, String.format("public static int %s=0x%08X;",
              javaId, Objects.hash(pkg, resourceType, javaId)));
        }
        idMatcher.appendTail(resourceIds);
        innerClassMatcher.appendReplacement(out,
            String.format("public static class %s {%s}", resourceType, resourceIds.toString()));
      }
      innerClassMatcher.appendTail(out);
      return out.toString();
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      if (file.getFileName().endsWith("R.java")) {
        byte[] content = Files.readAllBytes(file);
        if (staticIds) {
          content = replaceIdsWithStaticIds(UTF_8.decode(
              ByteBuffer.wrap(content)).toString()).getBytes(UTF_8);
        }
        ZipEntry entry = new ZipEntry(root.relativize(file).toString());

        entry.setMethod(ZipEntry.STORED);
        entry.setTime(ZIP_EPOCH);
        entry.setSize(content.length);
        CRC32 crc32 = new CRC32();
        crc32.update(content);
        entry.setCrc(crc32.getValue());
        zip.putNextEntry(entry);
        zip.write(content);
        zip.closeEntry();
      }
      return FileVisitResult.CONTINUE;
    }
  }

  private static final class ZipBuilderVisitor extends SimpleFileVisitor<Path> {
    // The earliest date representable in a zip file, 1-1-1980.
    private static final long ZIP_EPOCH = 315561600000L;
    private final ZipOutputStream zip;
    private final Path root;
    private final String directory;

    public ZipBuilderVisitor(ZipOutputStream zip, Path root, String directory) {
      this.zip = zip;
      this.root = root;
      this.directory = directory;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      byte[] content = Files.readAllBytes(file);

      CRC32 crc32 = new CRC32();
      crc32.update(content);

      ZipEntry entry = new ZipEntry(directory + "/" + root.relativize(file));
      entry.setMethod(ZipEntry.STORED);
      entry.setTime(ZIP_EPOCH);
      entry.setSize(content.length);
      entry.setCrc(crc32.getValue());

      zip.putNextEntry(entry);
      zip.write(content);
      zip.closeEntry();
      return FileVisitResult.CONTINUE;
    }
  }
}
