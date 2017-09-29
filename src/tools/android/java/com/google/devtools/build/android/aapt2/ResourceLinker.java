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
package com.google.devtools.build.android.aapt2;

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AaptCommandBuilder;
import com.google.devtools.build.android.AndroidResourceOutputs;
import com.google.devtools.build.android.Profiler;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/** Performs linking of {@link CompiledResources} using aapt2. */
public class ResourceLinker {

  /** Represents errors thrown during linking. */
  public static class LinkError extends RuntimeException {

    public LinkError(Throwable e) {
      super(e);
    }
  }

  private static Logger logger = Logger.getLogger(ResourceLinker.class.getName());

  private final Path aapt2;

  private final Path workingDirectory;

  private List<StaticLibrary> linkAgainst = ImmutableList.of();

  private Revision buildToolsVersion;
  private List<String> densities = ImmutableList.of();
  private Path androidJar;
  private Profiler profiler = Profiler.empty();
  private List<String> uncompressedExtensions = ImmutableList.of();
  private List<String> resourceConfigs = ImmutableList.of();
  private Path baseApk;
  private List<CompiledResources> include = ImmutableList.of();
  private List<Path> assetDirs = ImmutableList.of();

  private ResourceLinker(Path aapt2, Path workingDirectory) {
    this.aapt2 = aapt2;
    this.workingDirectory = workingDirectory;
  }

  public static ResourceLinker create(Path aapt2, Path workingDirectory) {
    Preconditions.checkArgument(Files.exists(workingDirectory));
    return new ResourceLinker(aapt2, workingDirectory);
  }

  public ResourceLinker profileUsing(Profiler profiler) {
    this.profiler = profiler;
    return this;
  }

  /** Dependent static libraries to be linked to. */
  public ResourceLinker dependencies(List<StaticLibrary> libraries) {
    this.linkAgainst = libraries;
    return this;
  }

  /** Dependent compiled resources to be included in the binary. */
  public ResourceLinker include(List<CompiledResources> include) {
    this.include = include;
    return this;
  }
  public ResourceLinker withAssets(List<Path> assetDirs) {
    this.assetDirs = assetDirs;
    return this;
  }

  public ResourceLinker buildVersion(Revision buildToolsVersion) {
    this.buildToolsVersion = buildToolsVersion;
    return this;
  }

  public ResourceLinker baseApkToLinkAgainst(Path baseApk) {
    this.baseApk = baseApk;
    return this;
  }

  public ResourceLinker filterToDensity(List<String> densities) {
    this.densities = densities;
    return this;
  }

  /**
   * Statically links the {@link CompiledResources} with the dependencies to produce a {@link
   * StaticLibrary}.
   *
   * @throws IOException
   */
  public StaticLibrary linkStatically(CompiledResources resources) {
    final Path outPath = workingDirectory.resolve("lib.ap_");
    final Path rTxt = workingDirectory.resolve("R.txt");
    final Path sourceJar = workingDirectory.resolve("r.srcjar");
    Path javaSourceDirectory = workingDirectory.resolve("java");

    try {
      profiler.startTask("linkstatic");
      logger.fine(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.LIBRARY)
              .add("link")
              .add("--manifest", resources.getManifest())
              .add("--static-lib")
              .add("--no-static-lib-packages")
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              .add("-R", resources.getZip())
              .addRepeated("-R",
                  include.stream()
                      .map(compiledResources -> compiledResources.getZip().toString())
                      .collect(Collectors.toList()))
              .addRepeated("-I", StaticLibrary.toPathStrings(linkAgainst))
              .add("--auto-add-overlay")
              .add("-o", outPath)
              .add("--java", javaSourceDirectory)
              .add("--output-text-symbols", rTxt)
              .execute(String.format("Statically linking %s", resources)));
      profiler.recordEndOf("linkstatic").startTask("sourcejar");
      AndroidResourceOutputs.createSrcJar(javaSourceDirectory, sourceJar, true /* staticIds */);
      profiler.recordEndOf("sourcejar");
      return StaticLibrary.from(outPath, rTxt, ImmutableList.of(), sourceJar);
    } catch (IOException e) {
      throw new LinkError(e);
    }
  }

  public PackagedResources link(CompiledResources compiled) {
    try {
      final Path outPath = workingDirectory.resolve("bin.apk");
      Path rTxt = workingDirectory.resolve("R.txt");
      Path proguardConfig = workingDirectory.resolve("proguard.cfg");
      Path mainDexProguard = workingDirectory.resolve("proguard.maindex.cfg");
      Path javaSourceDirectory = Files.createDirectories(workingDirectory.resolve("java"));
      Path resourceIds = workingDirectory.resolve("ids.txt");

      profiler.startTask("fulllink");
      logger.finer(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.DEFAULT)
              .add("link")
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              // Turn off namespaced resources
              .add("--no-static-lib-packages")
              .when(Objects.equals(logger.getLevel(), Level.FINE))
              .thenAdd("-v")
              .add("--manifest", compiled.getManifest())
              // Enables resource redefinition and merging
              .add("--auto-add-overlay")
              .when(densities.size() == 1)
              .thenAddRepeated("--preferred-density", densities)
              .add("--stable-ids", compiled.getStableIds())
              .addRepeated("-A",
                  assetDirs.stream().map(Path::toString).collect(Collectors.toList()))
              .addRepeated("-I", StaticLibrary.toPathStrings(linkAgainst))
              .addRepeated("-R",
                  include.stream()
                      .map(compiledResources -> compiledResources.getZip().toString())
                      .collect(Collectors.toList()))
              .add("-R", compiled.getZip())
              // Never compress apks.
              .add("-0", "apk")
              // Add custom no-compress extensions.
              .addRepeated("-0", uncompressedExtensions)
              // Filter by resource configuration type.
              .when(!resourceConfigs.isEmpty())
              .thenAdd("-c", Joiner.on(',').join(resourceConfigs))
              .add("--output-text-symbols", rTxt)
              .add("--emit-ids", resourceIds)
              .add("--java", javaSourceDirectory)
              .add("--proguard", proguardConfig)
              .add("--proguard-main-dex", mainDexProguard)
              .add("-o", outPath)
              .execute(String.format("Linking %s", compiled.getManifest())));
      profiler.recordEndOf("fulllink");
      profiler.startTask("optimize");
      if (densities.size() < 2) {
        return PackagedResources.of(
            outPath, rTxt, proguardConfig, mainDexProguard, javaSourceDirectory, resourceIds);
      }
      final Path optimized = workingDirectory.resolve("optimized.apk");
      logger.finer(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.DEFAULT)
              .add("optimize")
              .add("--target-densities", densities.stream().collect(Collectors.joining(",")))
              .add("-o", optimized)
              .add(outPath.toString())
              .execute(String.format("Optimizing %s", compiled.getManifest())));
      profiler.recordEndOf("optimize");
      return PackagedResources.of(
          optimized, rTxt, proguardConfig, mainDexProguard, javaSourceDirectory, resourceIds);
    } catch (IOException e) {
      throw new LinkError(e);
    }
  }

  public ResourceLinker storeUncompressed(List<String> uncompressedExtensions) {
    this.uncompressedExtensions = uncompressedExtensions;
    return this;
  }

  public ResourceLinker includeOnlyConfigs(List<String> resourceConfigs) {
    this.resourceConfigs = resourceConfigs;
    return this;
  }

  public ResourceLinker using(Path androidJar) {
    this.androidJar = androidJar;
    return this;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("aapt2", aapt2)
        .add("linkAgainst", linkAgainst)
        .add("buildToolsVersion", buildToolsVersion)
        .add("workingDirectory", workingDirectory)
        .add("densities", densities)
        .add("androidJar", androidJar)
        .add("uncompressedExtensions", uncompressedExtensions)
        .add("resourceConfigs", resourceConfigs)
        .add("baseApk", baseApk)
        .toString();
  }
}
