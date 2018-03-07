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

import static java.util.stream.Collectors.toList;

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AaptCommandBuilder;
import com.google.devtools.build.android.AndroidResourceOutputs;
import com.google.devtools.build.android.Profiler;
import com.google.devtools.build.android.ziputils.ZipIn;
import com.google.devtools.build.android.ziputils.ZipOut;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Performs linking of {@link CompiledResources} using aapt2. */
public class ResourceLinker {
  /** Represents errors thrown during linking. */
  public static class LinkError extends Aapt2Exception {

    private LinkError(Throwable e) {
      super(e);
    }

    public static LinkError of(Throwable e) {
      return new LinkError(e);
    }
  }

  private static Logger logger = Logger.getLogger(ResourceLinker.class.getName());

  private final Path aapt2;

  private final ListeningExecutorService executorService;
  private final Path workingDirectory;

  private List<StaticLibrary> linkAgainst = ImmutableList.of();

  private String customPackage;
  private boolean outputAsProto;

  private Revision buildToolsVersion;
  private List<String> densities = ImmutableList.of();
  private Path androidJar;
  private Profiler profiler = Profiler.empty();
  private List<String> uncompressedExtensions = ImmutableList.of();
  private List<String> resourceConfigs = ImmutableList.of();
  private Path baseApk;
  private List<CompiledResources> include = ImmutableList.of();
  private List<Path> assetDirs = ImmutableList.of();
  private boolean conditionalKeepRules = false;

  private ResourceLinker(
      Path aapt2, ListeningExecutorService executorService, Path workingDirectory) {
    this.aapt2 = aapt2;
    this.executorService = executorService;
    this.workingDirectory = workingDirectory;
  }

  public static ResourceLinker create(
      Path aapt2, ListeningExecutorService executorService, Path workingDirectory) {
    Preconditions.checkArgument(Files.exists(workingDirectory));
    return new ResourceLinker(aapt2, executorService, workingDirectory);
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

  public ResourceLinker conditionalKeepRules(boolean conditionalKeepRules) {
    this.conditionalKeepRules = conditionalKeepRules;
    return this;
  }

  public ResourceLinker baseApkToLinkAgainst(Path baseApk) {
    this.baseApk = baseApk;
    return this;
  }

  public ResourceLinker customPackage(String customPackage) {
    this.customPackage = customPackage;
    return this;
  }

  public ResourceLinker filterToDensity(List<String> densities) {
    this.densities = densities;
    return this;
  }

  public ResourceLinker outputAsProto(boolean outputAsProto) {
    this.outputAsProto = outputAsProto;
    return this;
  }

  /**
   * Statically links the {@link CompiledResources} with the dependencies to produce a {@link
   * StaticLibrary}.
   *
   * @throws IOException
   */
  public StaticLibrary linkStatically(CompiledResources compiled) {
    try {
      final Path outPath = workingDirectory.resolve("lib.apk");
      final Path rTxt = workingDirectory.resolve("R.txt");
      final Path sourceJar = workingDirectory.resolve("r.srcjar");
      Path javaSourceDirectory = workingDirectory.resolve("java");
      profiler.startTask("linkstatic");
      final Collection<String> pathsToLinkAgainst = StaticLibrary.toPathStrings(linkAgainst);
      logger.finer(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.LIBRARY)
              .add("link")
              .add("--manifest", compiled.getManifest())
              .add("--static-lib")
              .add("--no-static-lib-packages")
              .add("--custom-package", customPackage)
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              .when(outputAsProto)
              .thenAdd("--proto-format")
              .addParameterableRepeated("-R", compiledResourcesToPaths(compiled), workingDirectory)
              .addRepeated("-I", pathsToLinkAgainst)
              .add("--auto-add-overlay")
              .add("-o", outPath)
              .when(linkAgainst.size() == 1) // If using all compiled resources, generates sources
              .thenAdd("--java", javaSourceDirectory)
              .when(linkAgainst.size() == 1) // If using all compiled resources, generates R.txt
              .thenAdd("--output-text-symbols", rTxt)
              .execute(String.format("Statically linking %s", compiled)));
      profiler.recordEndOf("linkstatic");
      // working around aapt2 not producing transitive R.txt and R.java
      if (linkAgainst.size() > 1) {
        profiler.startTask("rfix");
        logger.finer(
            new AaptCommandBuilder(aapt2)
                .forBuildToolsVersion(buildToolsVersion)
                .forVariantType(VariantType.LIBRARY)
                .add("link")
                .add("--manifest", compiled.getManifest())
                .add("--no-static-lib-packages")
                .whenVersionIsAtLeast(new Revision(23))
                .thenAdd("--no-version-vectors")
                .when(outputAsProto)
                .thenAdd("--proto-format")
                // only link against jars
                .addRepeated(
                    "-I",
                    pathsToLinkAgainst.stream().filter(s -> s.endsWith(".jar")).collect(toList()))
                .add("-R", outPath)
                // only include non-jars
                .addRepeated(
                    "-R",
                    pathsToLinkAgainst.stream().filter(s -> !s.endsWith(".jar")).collect(toList()))
                .add("--auto-add-overlay")
                .add("-o", outPath.resolveSibling("transitive.apk"))
                .add("--java", javaSourceDirectory)
                .add("--output-text-symbols", rTxt)
                .execute(String.format("Generating R files %s", compiled)));
        profiler.recordEndOf("rfix");
      }

      profiler.startTask("sourcejar");
      AndroidResourceOutputs.createSrcJar(javaSourceDirectory, sourceJar, true /* staticIds */);
      profiler.recordEndOf("sourcejar");
      return StaticLibrary.from(outPath, rTxt, ImmutableList.of(), sourceJar);
    } catch (IOException e) {
      throw LinkError.of(e);
    }
  }

  private List<String> compiledResourcesToPaths(CompiledResources compiled) throws IOException {
    // Using sequential streams to maintain the overlay order for aapt2.
    return Stream.concat(include.stream(), Stream.of(compiled))
        .sequential()
        .map(CompiledResources::getZip)
        .map(z -> executorService.submit(() -> filterZip(z)))
        .map(rethrowLinkError(Future::get))
        // the process will always take as long as the longest Future
        .map(Path::toString)
        .collect(toList());
  }

  private Path filterZip(Path path) throws IOException {
    Path outPath =
        workingDirectory
            .resolve("filtered")
            // make absolute paths relative so that resolve will make a new path.
            .resolve(path.isAbsolute() ? path.subpath(1, path.getNameCount()) : path);
    // TODO(74258184): How can this path already exist?
    if (Files.exists(outPath)) {
      return outPath;
    }
    Files.createDirectories(outPath.getParent());
    try (FileChannel inChannel = FileChannel.open(path, StandardOpenOption.READ);
        FileChannel outChannel =
            FileChannel.open(outPath, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      final ZipIn zipIn = new ZipIn(inChannel, path.toString());
      final ZipOut zipOut = new ZipOut(outChannel, outPath.toString());
      zipIn.scanEntries(
          (in, header, dirEntry, data) -> {
            if (header.getFilename().endsWith(".flat")) {
              zipOut.nextEntry(dirEntry);
              zipOut.write(header);
              zipOut.write(data);
            }
          });
      zipOut.close();
    }
    return outPath;
  }

  private static <T, R> Function<T, R> rethrowLinkError(CheckedFunction<T, R> checked) {
    return (T arg) -> {
      try {
        return checked.apply(arg);
      } catch (ExecutionException e) {
        throw LinkError.of(Optional.ofNullable(e.getCause()).orElse(e)); // unwrap
      } catch (IOException e) {
        throw LinkError.of(e);
      } catch (Throwable e) { // unexpected error, rethrow for debugging.
        throw new RuntimeException(e);
      }
    };
  }

  @FunctionalInterface
  private interface CheckedFunction<T, R> {
    R apply(T arg) throws Throwable;
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
              .when(outputAsProto)
              .thenAdd("--proto-format")
              .when(Objects.equals(logger.getLevel(), Level.FINE))
              .thenAdd("-v")
              .add("--manifest", compiled.getManifest())
              // Enables resource redefinition and merging
              .add("--auto-add-overlay")
              .add("--custom-package", customPackage)
              .when(densities.size() == 1)
              .thenAddRepeated("--preferred-density", densities)
              .add("--stable-ids", compiled.getStableIds())
              .addRepeated(
                  "-A",
                  Streams.concat(
                          assetDirs.stream().map(Path::toString),
                          compiled.getAssetsStrings().stream())
                      .collect(toList()))
              .addRepeated("-I", StaticLibrary.toPathStrings(linkAgainst))
              .addParameterableRepeated("-R", compiledResourcesToPaths(compiled), workingDirectory)
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
              .when(conditionalKeepRules)
              .thenAdd("--proguard-conditional-keep-rules")
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
