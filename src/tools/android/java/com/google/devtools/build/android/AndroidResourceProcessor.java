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

import com.android.annotations.Nullable;
import com.android.builder.core.VariantConfiguration;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.builder.internal.SymbolLoader;
import com.android.builder.internal.SymbolWriter;
import com.android.builder.model.AaptOptions;
import com.android.ide.common.internal.CommandLineRunner;
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
import com.android.utils.StdLogger;

import org.xml.sax.SAXException;

import java.io.File;
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
      Files.setLastModifiedTime(srcJar, FileTime.fromMillis(0L));
    } catch (IOException e) {
      Throwables.propagate(e);
    }
  }

  /**
   * Processes resources for generated sources, configs and packaging resources.
   */
  public void processResources(
      Path aapt,
      Path androidJar,
      VariantConfiguration.Type variantType,
      boolean debug,
      String customPackageForR,
      AaptOptions aaptOptions,
      Collection<String> resourceConfigs,
      String applicationId,
      int versionCode,
      String versionName,
      MergedAndroidData primaryData,
      List<DependencyAndroidData> dependencyData,
      Path workingDirectory,
      Path sourceOut,
      Path packageOut,
      Path proguardOut,
      Path manifestOut) throws IOException, InterruptedException, LoggedErrorException {
    List<SymbolFileProvider> libraries = new ArrayList<>();
    List<String> packages = new ArrayList<>();
    for (DependencyAndroidData dataDep : dependencyData) {
      SymbolFileProvider library = dataDep.asSymbolFileProvider();
      libraries.add(library);
      packages.add(VariantConfiguration.getManifestPackage(library.getManifest()));
    }

    Path androidManifest = processManifest(

        variantType == VariantConfiguration.Type.DEFAULT ? applicationId : customPackageForR,
        versionCode,
        versionName,
        primaryData,
        workingDirectory,
        variantType == VariantConfiguration.Type.DEFAULT
            ? ManifestMerger2.MergeType.APPLICATION : ManifestMerger2.MergeType.LIBRARY);

    File resFolder = primaryData.getResourceDirFile();
    File assetsDir = primaryData.getAssetDirFile();

    List<String> command = new ArrayList<>();
    
    command.add(aapt.toString());
    command.add("package");
    
    // Trigger the aapt logging level on the Logger.
    if (stdLogger.getLevel() == StdLogger.Level.VERBOSE) {
        command.add("-v");
    }
    
    // Overwrite existing files, if they exist.
    command.add("-f");
    
    // Resources are precrunched in the merge process.
    command.add("--no-crunch");

    // Do not automatically generate versioned copies of vector XML resources.
    command.add("--no-version-vectors");

    // Add the android.jar as a base input.
    command.add("-I");
    command.add(androidJar.toString());

    // Add the manifest for validation.
    command.add("-M");
    command.add(androidManifest.toAbsolutePath().toString());

    if (resFolder.isDirectory()) {
      command.add("-S");
      command.add(resFolder.getAbsolutePath());
    }

    if (assetsDir != null && assetsDir.isDirectory()) {
      command.add("-A");
      command.add(assetsDir.getAbsolutePath());
    }

    // Outputs
    if (sourceOut != null) {
      prepareOutputPath(sourceOut);
      command.add("-m");
      command.add("-J");
      command.add(sourceOut.toString());
      command.add("--output-text-symbols");
      command.add(sourceOut.toString());
    }

    if (packageOut != null) {
      command.add("-F");
      command.add(packageOut.toString());
    }

    if (proguardOut != null) {
      command.add("-G");
      command.add(proguardOut.toString());
    }
    
    // Additional options.
    if (debug) {
      command.add("--debug-mode");
    }

    if (customPackageForR != null) {
      command.add("--custom-package");
      command.add(customPackageForR);
      stdLogger.verbose("Custom package for R class: '%s'", customPackageForR);
    }

    // If it is a library, do not generate final java ids.
    if (variantType == VariantConfiguration.Type.LIBRARY) {
      command.add("--non-constant-id");
    }

    if (variantType == VariantConfiguration.Type.DEFAULT) {
      // Generate the dependent R and Manifest files.
      command.add("--extra-packages");
      command.add(Joiner.on(":").join(packages));
    }

    if (aaptOptions.getIgnoreAssets() != null) {
      command.add("--ignore-assets");
      command.add(aaptOptions.getIgnoreAssets());
    }

    if (aaptOptions.getFailOnMissingConfigEntry()) {
      command.add("--error-on-missing-config-entry");
    }

    // Never compress apks.
    command.add("-0");
    command.add("apk");

    // Add custom no-compress extensions.
    for (String noCompress : aaptOptions.getNoCompress()) {
      command.add("-0");
      command.add(noCompress);
    }

    // Filter by resource configuration type.
    if (!resourceConfigs.isEmpty()) {
      command.add("-c");
      command.add(Joiner.on(',').join(resourceConfigs));
    }

    new CommandLineRunner(stdLogger).runCmdLine(command, null);

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
    if (manifestOut != null) {
      Files.copy(androidManifest, manifestOut);

      Files.setLastModifiedTime(manifestOut, FileTime.fromMillis(0L));
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

  private Path processManifest(

      String newManifestPackage,
      int versionCode,
      String versionName,
      MergedAndroidData primaryData,
      Path workingDirectory,
      ManifestMerger2.MergeType mergeType) throws IOException {
    if (versionCode != -1 || versionName != null || newManifestPackage != null) {
      Path androidManifest =
          Files.createDirectories(workingDirectory).resolve("AndroidManifest.xml");

      // The generics on Invoker don't make sense, so ignore them.
      @SuppressWarnings("unchecked")
      Invoker<?> manifestMergerInvoker =
          ManifestMerger2.newMerger(primaryData.getManifestFile(), stdLogger, mergeType);
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
            writeMergedManifest(mergingReport, androidManifest);
            break;
          case SUCCESS:
            writeMergedManifest(mergingReport, androidManifest);
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
      return androidManifest;
    }
    return primaryData.getManifestFile().toPath();
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

  private String prepareOutputPath(@Nullable Path out) throws IOException {
    if (out == null) {
      return null;
    }
    return Files.createDirectories(out).toString();
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
}

