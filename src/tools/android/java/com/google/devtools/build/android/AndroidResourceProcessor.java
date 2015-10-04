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

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import com.android.annotations.Nullable;
import com.android.builder.core.AndroidBuilder;
import com.android.builder.core.VariantConfiguration;
import com.android.builder.dependency.ManifestDependency;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.builder.model.AaptOptions;
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
import com.android.utils.StdLogger;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Provides a wrapper around the AOSP build tools for resource processing.
 */
public class AndroidResourceProcessor {
  private final StdLogger stdLogger;

  public AndroidResourceProcessor(StdLogger stdLogger) {
    this.stdLogger = stdLogger;
  }

  /**
   * Copies the R.txt to the expected place.
   */
  public void copyRToOutput(Path generatedSourceRoot, Path rOutput) {
    try {
      Files.createDirectories(rOutput.getParent());
      final Path source = generatedSourceRoot.resolve("R.txt");
      if (Files.exists(source)) {
        Files.copy(source, rOutput);
      } else {
        // The R.txt wasn't generated, create one for future inheritance, as Bazel always requires
        // outputs. This state occurs when there are no resource directories.
        Files.createFile(rOutput);
      }
    } catch (IOException e) {
      Throwables.propagate(e);
    }
  }

  /**
   * Creates a zip archive from all found R.java files.
   */
  public void createSrcJar(Path generatedSourcesRoot, Path srcJar) {
    try {
      Files.createDirectories(srcJar.getParent());
      try (final ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(srcJar))) {
        Files.walkFileTree(generatedSourcesRoot,
            new SymbolFileSrcJarBuildingVisitor(zip, generatedSourcesRoot));
      }
    } catch (IOException e) {
      Throwables.propagate(e);
    }
  }

  /**
   * Processes resources for generated sources, configs and packaging resources.
   */
  public void processResources(
      AndroidBuilder builder,
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
      @Nullable Path sourceOut,
      @Nullable Path packageOut,
      @Nullable Path proguardOut) throws IOException, InterruptedException, LoggedErrorException {
    ImmutableList.Builder<SymbolFileProvider> libraries = ImmutableList.builder();
    for (DependencyAndroidData dataDep : dependencyData) {
      libraries.add(dataDep.asSymbolFileProvider());
    }

    File androidManifest = processManifest(
        applicationId,
        versionCode,
        versionName,
        primaryData,
        workingDirectory,
        builder);

    builder.processResources(
        androidManifest,
        primaryData.getResourceDirFile(),
        primaryData.getAssetDirFile(),
        libraries.build(),
        customPackageForR,
        prepareOutputPath(sourceOut),
        prepareOutputPath(sourceOut),
        packageOut != null ? packageOut.toString() : null,
        proguardOut != null ? proguardOut.toString() : null,
        variantType,
        debug,
        aaptOptions,
        resourceConfigs,
        true // boolean enforceUniquePackageName
        );
  }

  private File processManifest(
      String applicationId,
      int versionCode,
      String versionName,
      MergedAndroidData primaryData,
      Path workingDirectory,
      AndroidBuilder builder) throws IOException {
    if (versionCode != -1 || versionName != null || applicationId != null) {
      Path androidManifest =
          Files.createDirectories(workingDirectory).resolve("AndroidManifest.xml");
      // stamp version and applicationId (if provided) into the manifest
      builder.mergeManifests(
          primaryData.getManifestFile(), // mainManifest,
          ImmutableList.<File>of(),
          ImmutableList.<ManifestDependency>of(),
          applicationId,
          versionCode,
          versionName,
          null, // String minSdkVersion
          null, // String targetSdkVersion
          null, // int maxSdkVersion
          androidManifest.toString(),
          ManifestMerger2.MergeType.APPLICATION,
          ImmutableMap.<String, String>of());
      return androidManifest.toFile();
    }
    return primaryData.getManifestFile();
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
  private final class SymbolFileSrcJarBuildingVisitor extends SimpleFileVisitor<Path> {

    // The earliest date representable in a zip file, 1-1-1980.
    private static final long ZIP_EPOCH = 315561600000L;
    private final ZipOutputStream zip;
    private final Path root;

    private SymbolFileSrcJarBuildingVisitor(ZipOutputStream zip, Path root) {
      this.zip = zip;
      this.root = root;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      if (file.getFileName().endsWith("R.java")) {
        byte[] content = Files.readAllBytes(file);
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
