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

import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTCRC;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTLEN;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENCRC;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCFLG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCTIM;
import static java.util.stream.Collectors.toList;

import com.android.builder.core.VariantConfiguration;
import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AaptCommandBuilder;
import com.google.devtools.build.android.AndroidCompiledDataDeserializer;
import com.google.devtools.build.android.AndroidResourceOutputs;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.Profiler;
import com.google.devtools.build.android.ResourceProcessorBusyBox;
import com.google.devtools.build.android.aapt2.ResourceCompiler.CompiledType;
import com.google.devtools.build.android.ziputils.DataDescriptor;
import com.google.devtools.build.android.ziputils.DirectoryEntry;
import com.google.devtools.build.android.ziputils.DosTime;
import com.google.devtools.build.android.ziputils.EntryHandler;
import com.google.devtools.build.android.ziputils.LocalFileHeader;
import com.google.devtools.build.android.ziputils.ZipIn;
import com.google.devtools.build.android.ziputils.ZipOut;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Performs linking of {@link CompiledResources} using aapt2. */
public class ResourceLinker {

  private static final Predicate<String> IS_JAR = s -> s.endsWith(".jar");

  /**
   * A file extension to indicate whether an apk is a proto or binary format.
   *
   * <p>The file extension is tremendously important to aapt2 -- it uses it determine how to
   * interpret the contents of the file.
   */
  public static final String PROTO_EXTENSION = "-pb.apk";

  private boolean debug;
  private static final Predicate<DirectoryEntry> IS_FLAT_FILE =
      h -> h.getFilename().endsWith(".flat");

  private static final Predicate<DirectoryEntry> COMMENT_ABSENT =
      h -> Strings.isNullOrEmpty(h.getComment());

  private static final Predicate<DirectoryEntry> USE_GENERATED =
      COMMENT_ABSENT.or(
          h -> ResourceCompiler.getCompiledType(h.getFilename()) == CompiledType.GENERATED);

  private static final Predicate<DirectoryEntry> USE_DEFAULT =
      COMMENT_ABSENT.or(
          h -> ResourceCompiler.getCompiledType(h.getComment()) != CompiledType.GENERATED);

  private static final ImmutableSet<String> PSEUDO_LOCALE_FILTERS =
      ImmutableSet.of("en_XA", "ar_XB");

  private static final boolean OVERRIDE_STYLES_INSTEAD_OF_OVERLAYING =
      ResourceProcessorBusyBox.getProperty("override_styles_instead_of_overlaying");

  /** Represents errors thrown during linking. */
  public static class LinkError extends Aapt2Exception {

    private LinkError(Throwable e) {
      super(e);
    }

    public static LinkError of(Throwable e) {
      return new LinkError(e);
    }
  }

  private boolean generatePseudoLocale;

  private static Logger logger = Logger.getLogger(ResourceLinker.class.getName());

  private final Path aapt2;

  private final ListeningExecutorService executorService;
  private final Path workingDirectory;

  private List<StaticLibrary> linkAgainst = ImmutableList.of();

  private String customPackage;
  private boolean outputAsProto;

  private Revision buildToolsVersion;
  private List<String> densities = ImmutableList.of();
  private Profiler profiler = Profiler.empty();
  private List<String> uncompressedExtensions = ImmutableList.of();
  private List<String> resourceConfigs = ImmutableList.of();
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

  public ResourceLinker includeGeneratedLocales(boolean generatePseudoLocale) {
    this.generatePseudoLocale = generatePseudoLocale;
    return this;
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

  public ResourceLinker debug(boolean debug) {
    this.debug = debug;
    return this;
  }

  public ResourceLinker conditionalKeepRules(boolean conditionalKeepRules) {
    this.conditionalKeepRules = conditionalKeepRules;
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
              .when(outputAsProto) // Used for testing: aapt2 does not output static libraries in
              // proto format.
              .thenAdd("--proto-format")
              .when(!outputAsProto)
              .thenAdd("--static-lib")
              .add("--manifest", compiled.getManifest())
              .add("--no-static-lib-packages")
              .add("--custom-package", customPackage)
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              .addParameterableRepeated(
                  "-R", compiledResourcesToPaths(compiled, IS_FLAT_FILE), workingDirectory)
              .addRepeated("-I", pathsToLinkAgainst)
              .add("--auto-add-overlay")
              .when(OVERRIDE_STYLES_INSTEAD_OF_OVERLAYING)
              .thenAdd("--override-styles-instead-of-overlaying")
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
                .addRepeated("-I", pathsToLinkAgainst.stream().filter(IS_JAR).collect(toList()))
                .add("-R", outPath)
                // only include non-jars
                .addRepeated(
                    "-R", pathsToLinkAgainst.stream().filter(IS_JAR.negate()).collect(toList()))
                .add("--auto-add-overlay")
                .when(OVERRIDE_STYLES_INSTEAD_OF_OVERLAYING)
                .thenAdd("--override-styles-instead-of-overlaying")
                .add("-o", outPath.resolveSibling("transitive.apk"))
                .add("--java", javaSourceDirectory)
                .add("--output-text-symbols", rTxt)
                .execute(String.format("Generating R files %s", compiled)));
        profiler.recordEndOf("rfix");
      }

      profiler.startTask("sourcejar");
      AndroidResourceOutputs.createSrcJar(javaSourceDirectory, sourceJar, /* staticIds= */ true);
      profiler.recordEndOf("sourcejar");
      return StaticLibrary.from(outPath, rTxt, ImmutableList.of(), sourceJar);
    } catch (IOException e) {
      throw LinkError.of(e);
    }
  }

  private List<String> compiledResourcesToPaths(
      CompiledResources compiled, Predicate<DirectoryEntry> shouldKeep) {
    // NB: "include" can have duplicates, in particular because Aapt2ResourcePackagingAction
    // creates this by concatenating two different options.  Since the *last* definition of anything
    // takes precedence, keep the last instance of each entry.
    List<Path> dedupedZips =
        Stream.concat(include.stream(), Stream.of(compiled))
            .map(CompiledResources::getZip)
            .collect(ImmutableList.toImmutableList())
            .reverse()
            .stream()
            .distinct()
            .collect(ImmutableList.toImmutableList())
            .reverse();

    return dedupedZips.stream()
        .map(z -> executorService.submit(() -> filterZip(z, shouldKeep)))
        .map(rethrowLinkError(Future::get))
        // the process will always take as long as the longest Future
        .map(Path::toString)
        .collect(toList());
  }

  private Path filterZip(Path path, Predicate<DirectoryEntry> shouldKeep) throws IOException {
    Path outPath =
        workingDirectory
            .resolve("filtered")
            // make absolute paths relative so that resolve will make a new path.
            .resolve(path.isAbsolute() ? path.subpath(1, path.getNameCount()) : path);
    Files.createDirectories(outPath.getParent());
    try (FileChannel inChannel = FileChannel.open(path, StandardOpenOption.READ);
        FileChannel outChannel =
            FileChannel.open(outPath, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      final ZipIn zipIn = new ZipIn(inChannel, path.toString());
      final ZipOut zipOut = new ZipOut(outChannel, outPath.toString());
      zipIn.scanEntries(
          (in, header, dirEntry, data) -> {
            if (shouldKeep.test(dirEntry)) {
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

  private String replaceExtension(String fileName, String newExtension) {
    int lastIndex = fileName.lastIndexOf('.');
    if (lastIndex == -1) {
      return fileName.concat(".").concat(newExtension);
    }
    return fileName.substring(0, lastIndex).concat(".").concat(newExtension);
  }

  private ProtoApk linkProtoApk(
      CompiledResources compiled,
      Path rTxt,
      Path proguardConfig,
      Path mainDexProguard,
      Path javaSourceDirectory,
      Path resourceIds)
      throws IOException {
    profiler.startTask("fulllink");
    final Path linked = workingDirectory.resolve("bin." + PROTO_EXTENSION);
    logger.fine(
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
            .when(OVERRIDE_STYLES_INSTEAD_OF_OVERLAYING)
            .thenAdd("--override-styles-instead-of-overlaying")
            // Always link to proto, as resource shrinking needs the extra information.
            .add("--proto-format")
            .when(debug)
            .thenAdd("--debug-mode")
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
            .addParameterableRepeated(
                "-R",
                compiledResourcesToPaths(
                    compiled,
                    generatePseudoLocale
                            && resourceConfigs.stream().anyMatch(PSEUDO_LOCALE_FILTERS::contains)
                        ? IS_FLAT_FILE.and(USE_GENERATED)
                        : IS_FLAT_FILE.and(USE_DEFAULT)),
                workingDirectory)
            // Never compress apks.
            .add("-0", ".apk")
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
            .add("-o", linked)
            .execute(String.format("Linking %s", compiled.getManifest())));
    profiler.recordEndOf("fulllink");
    return ProtoApk.readFrom(optimize(compiled, linked));
  }

  private Path combineApks(Path protoApk, Path binaryApk, Path workingDirectory)
      throws IOException {
    // Linking against apk as a static library elides assets, among other things.
    // So, copy the missing details to the new apk.
    profiler.startTask("combine");
    final Path combined = workingDirectory.resolve("combined.apk");
    try (FileChannel nonResourceChannel = FileChannel.open(protoApk, StandardOpenOption.READ);
        FileChannel resourceChannel = FileChannel.open(binaryApk, StandardOpenOption.READ);
        FileChannel outChannel =
            FileChannel.open(combined, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      final ZipIn resourcesIn = new ZipIn(resourceChannel, binaryApk.toString());
      final ZipIn nonResourcesIn = new ZipIn(nonResourceChannel, protoApk.toString());
      final ZipOut zipOut = new ZipOut(outChannel, combined.toString());

      Set<String> skip = new LinkedHashSet<>();
      skip.add("resources.pb");
      final EntryHandler entryHandler =
          (in, header, dirEntry, data) -> {
            final String filename = dirEntry.getFilename();
            // Make sure we aren't copying the same entry twice.
            if (!skip.contains(filename)) {
              skip.add(filename);
              String comment = dirEntry.getComment();
              byte[] extra = dirEntry.getExtraData();
              zipOut.nextEntry(
                  dirEntry.clone(filename, extra, comment).set(CENTIM, DosTime.EPOCHISH.time));
              zipOut.write(header.clone(filename, extra).set(LOCTIM, DosTime.EPOCHISH.time));
              zipOut.write(data);
              if ((header.get(LOCFLG) & LocalFileHeader.SIZE_MASKED_FLAG) != 0) {
                DataDescriptor desc =
                    DataDescriptor.allocate()
                        .set(EXTCRC, dirEntry.get(CENCRC))
                        .set(EXTSIZ, dirEntry.get(CENSIZ))
                        .set(EXTLEN, dirEntry.get(CENLEN));
                zipOut.write(desc);
              }
            }
          };
      resourcesIn.scanEntries(entryHandler);
      nonResourcesIn.scanEntries(entryHandler);
      zipOut.close();
      return combined;
    } finally {
      profiler.recordEndOf("combine");
    }
  }

  private Path extractPackages(CompiledResources compiled) throws IOException {
    Path packages = workingDirectory.resolve("packages");
    try (BufferedWriter writer = Files.newBufferedWriter(packages, StandardOpenOption.CREATE_NEW)) {
      for (CompiledResources resources : FluentIterable.from(include).append(compiled)) {
        writer.append(VariantConfiguration.getManifestPackage(resources.getManifest().toFile()));
        writer.newLine();
      }
    }
    return packages;
  }

  private Path extractAttributes(CompiledResources compiled) throws IOException {
    profiler.startTask("attributes");
    Path attributes = workingDirectory.resolve("tool.attributes");
    // extract tool annotations from the compile resources.
    final SdkToolAttributeWriter writer = new SdkToolAttributeWriter(attributes);
    for (CompiledResources resources : FluentIterable.from(include).append(compiled)) {
      AndroidCompiledDataDeserializer.readAttributes(resources)
          .forEach((key, value) -> value.writeResource((FullyQualifiedName) key, writer));
    }
    writer.flush();
    profiler.recordEndOf("attributes");
    return attributes;
  }

  private Path optimize(CompiledResources compiled, Path binary) throws IOException {
    if (densities.size() < 2) {
      return binary;
    }

    profiler.startTask("optimize");
    final Path optimized = workingDirectory.resolve("optimized." + PROTO_EXTENSION);
    logger.fine(
        new AaptCommandBuilder(aapt2)
            .forBuildToolsVersion(buildToolsVersion)
            .forVariantType(VariantType.DEFAULT)
            .add("optimize")
            .when(Objects.equals(logger.getLevel(), Level.FINE))
            .thenAdd("-v")
            // TODO(b/138166830): Simplify behavior specific to number of densities. There's likely
            // little to lose in passing a single-element density list, which we would confirm in
            // the APK analyzer dashboard.
            .when(densities.size() >= 2)
            .thenAdd("--target-densities", densities.stream().collect(Collectors.joining(",")))
            .add("-o", optimized)
            .add(binary.toString())
            .execute(String.format("Optimizing %s", compiled.getManifest())));
    profiler.recordEndOf("optimize");
    return optimized;
  }

  /** Links compiled resources into an apk */
  public PackagedResources link(CompiledResources compiled) {
    try {
      Path rTxt = workingDirectory.resolve("R.txt");
      Path proguardConfig = workingDirectory.resolve("proguard.cfg");
      Path mainDexProguard = workingDirectory.resolve("proguard.maindex.cfg");
      Path javaSourceDirectory = Files.createDirectories(workingDirectory.resolve("java"));
      Path resourceIds = workingDirectory.resolve("ids.txt");
      try (ProtoApk protoApk =
          linkProtoApk(
              compiled, rTxt, proguardConfig, mainDexProguard, javaSourceDirectory, resourceIds)) {
        return PackagedResources.of(
            outputAsProto
                ? protoApk.asApkPath()
                : link(protoApk, resourceIds), // convert proto to binary
            protoApk.asApkPath(),
            rTxt,
            proguardConfig,
            mainDexProguard,
            javaSourceDirectory,
            resourceIds,
            extractAttributes(compiled),
            extractPackages(compiled));
      }

    } catch (IOException e) {
      throw new LinkError(e);
    }
  }

  /** Link a proto apk to produce an apk. */
  // aapt2 has a "convert" command which has similar behavior, but at the time of writing, will
  // not compress any files in the res/ directory.
  public Path link(ProtoApk protoApk, Path resourceIds) {
    try {
      final Path protoApkPath = protoApk.asApkPath();
      final Path working =
          workingDirectory
              .resolve("link-proto")
              .resolve(replaceExtension(protoApkPath.getFileName().toString(), "working"));
      final Path manifest = protoApk.writeManifestAsXmlTo(working);
      final Path apk = working.resolve("binary.apk");
      logger.fine(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.DEFAULT)
              .add("link")
              .when(Objects.equals(logger.getLevel(), Level.FINE))
              .thenAdd("-v")
              .whenVersionIsAtLeast(new Revision(23))
              .thenAdd("--no-version-vectors")
              .add("--stable-ids", resourceIds)
              .add("--manifest", manifest)
              .addRepeated("-I", StaticLibrary.toPathStrings(linkAgainst))
              .add("-R", protoApk.asApkPath())
              .add("-o", apk.toString())
              .execute(String.format("Re-linking %s", protoApkPath)));
      return combineApks(protoApkPath, apk, working);
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

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("aapt2", aapt2)
        .add("linkAgainst", linkAgainst)
        .add("buildToolsVersion", buildToolsVersion)
        .add("workingDirectory", workingDirectory)
        .add("densities", densities)
        .add("uncompressedExtensions", uncompressedExtensions)
        .add("resourceConfigs", resourceConfigs)
        .toString();
  }
}
