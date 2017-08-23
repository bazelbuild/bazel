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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.AaptCommandBuilder;
import com.google.devtools.build.android.AndroidResourceOutputs;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.zip.ZipFile;

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
  private String density;
  private Path androidJar;
  private List<String> uncompressedExtensions = ImmutableList.of();
  private List<String> resourceConfigs = ImmutableList.of();
  private Path baseApk;
  private List<StaticLibrary> include;

  private ResourceLinker(Path aapt2, Path workingDirectory) {
    this.aapt2 = aapt2;
    this.workingDirectory = workingDirectory;
  }

  public static ResourceLinker create(Path aapt2, Path workingDirectory) {
    return new ResourceLinker(aapt2, workingDirectory);
  }


  /** Dependent static libraries to be linked to. */
  public ResourceLinker dependencies(List<StaticLibrary> libraries) {
    this.linkAgainst = libraries;
    return this;
  }

  /** Dependent static libraries to be included in the binary. */
  public ResourceLinker include(List<StaticLibrary> include) {
    this.include = include;
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

  public ResourceLinker filterToDensity(List<String> densitiesToFilter) {
    if (densitiesToFilter.size() > 1) {
      logger.warning("Multiple densities not yet supported with aapt2");
    } else if (densitiesToFilter.size() > 0) {
      density = Iterables.getOnlyElement(densitiesToFilter);
    }
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
              .addRepeated("-R", unzipCompiledResources(resources.getZip()))
              .addRepeated("-I", StaticLibrary.toPathStrings(linkAgainst))
              .add("--java", javaSourceDirectory)
              .add("--auto-add-overlay")
              .add("-o", outPath)
              .add("--java", javaSourceDirectory)
              .add("--output-text-symbols", rTxt)
              .execute(String.format("Statically linking %s", resources)));

      AndroidResourceOutputs.createSrcJar(javaSourceDirectory, sourceJar, true /* staticIds */);

      return StaticLibrary.from(outPath, rTxt, ImmutableList.of(), sourceJar);
    } catch (IOException e) {
      throw new LinkError(e);
    }
  }

  public PackagedResources link(CompiledResources compiled) {
    final Path outPath = workingDirectory.resolve("bin.ap_");
    Path rTxt = workingDirectory.resolve("R.txt");
    Path proguardConfig = workingDirectory.resolve("proguard.cfg");
    Path mainDexProguard = workingDirectory.resolve("proguard.maindex.cfg");
    Path javaSourceDirectory = workingDirectory.resolve("java");

    try {
      logger.fine(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.DEFAULT)
              .add("link")
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              .add("--no-static-lib-packages")
              .when(logger.getLevel() == Level.FINE)
              .thenAdd("-v")
              .add("--manifest", compiled.getManifest())
              .add("--auto-add-overlay")
              .addRepeated("-A", compiled.getAssetsStrings())
              .addRepeated("-I", StaticLibrary.toPathStrings(linkAgainst))
              .addRepeated("-R", StaticLibrary.toPathStrings(include))
              .addRepeated("-R", unzipCompiledResources(compiled.getZip()))
              // Never compress apks.
              .add("-0", "apk")
              // Add custom no-compress extensions.
              .addRepeated("-0", uncompressedExtensions)
              .addRepeated("-A", StaticLibrary.toAssetPaths(include))
              .when(density != null)
              .thenAdd("--preferred-density", density)
              // Filter by resource configuration type.
              .when(!resourceConfigs.isEmpty())
              .thenAdd("-c", Joiner.on(',').join(resourceConfigs))
              .add("--output-text-symbols", rTxt)
              .add("--java", javaSourceDirectory)
              .add("--proguard", proguardConfig)
              .add("--proguard-main-dex", mainDexProguard)
              .add("-o", outPath)
              .execute(String.format("Linking %s", compiled.getManifest())));
      return PackagedResources.of(
          outPath, rTxt, proguardConfig, mainDexProguard, javaSourceDirectory);
    } catch (IOException e) {
      throw new LinkError(e);
    }
  }

  private List<String> unzipCompiledResources(Path resourceZip) throws IOException {
    final ZipFile zipFile = new ZipFile(resourceZip.toFile());
    return zipFile
        .stream()
        .map(
            entry -> {
              final Path resolve = workingDirectory.resolve(entry.getName());
              try {
                Files.createDirectories(resolve.getParent());
                return Files.write(resolve, ByteStreams.toByteArray(zipFile.getInputStream(entry)));
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
            })
        .map(Path::toString)
        .collect(Collectors.toList());
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
        .add("density", density)
        .add("androidJar", androidJar)
        .add("uncompressedExtensions", uncompressedExtensions)
        .add("resourceConfigs", resourceConfigs)
        .add("baseApk", baseApk)
        .toString();
  }
}
