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
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENHOW;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCFLG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCHOW;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCSIZ;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCTIM;
import static java.util.stream.Collectors.toList;
import static java.util.zip.ZipEntry.DEFLATED;
import static java.util.zip.ZipEntry.STORED;

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
import com.google.common.io.ByteSource;
import com.google.common.io.ByteStreams;
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
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
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
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;

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
  private Optional<Integer> packageId = Optional.empty();
  private boolean outputAsProto;

  private Revision buildToolsVersion;
  private List<String> densities = ImmutableList.of();
  private Profiler profiler = Profiler.empty();
  private List<String> uncompressedExtensions = ImmutableList.of();
  private List<String> resourceConfigs = ImmutableList.of();
  private List<CompiledResources> include = ImmutableList.of();
  private List<Path> assetDirs = ImmutableList.of();
  private boolean conditionalKeepRules = false;
  private boolean includeProguardLocationReferences = false;

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

  public ResourceLinker packageId(Optional<Integer> packageId) {
    this.packageId = packageId;
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
            .when(packageId.isPresent())
            .thenAdd("--package-id", "0x" + Integer.toHexString(packageId.orElse(0x7f)))
            .when(packageId.map(id -> id < 0x7f).orElse(false))
            .thenAdd("--allow-reserved-package-id")
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
            // Add custom no-compress extensions. This ultimately doesn't matter - these files
            // may be compressed during a later intermediate step, but will be decompressed again
            // during final APK generation, in the native android_binary rule.
            .addRepeated("-0", uncompressedExtensions)
            // Filter by resource configuration type.
            .when(!resourceConfigs.isEmpty())
            .thenAdd("-c", Joiner.on(',').join(resourceConfigs))
            .add("--output-text-symbols", rTxt)
            .add("--emit-ids", resourceIds)
            .add("--java", javaSourceDirectory)
            .add("--proguard", proguardConfig)
            .add("--proguard-main-dex", mainDexProguard)
            // By default, exclude the file path location comments, since the paths
            // include temporary directory names, which otherwise cause
            // nondeterministic build output.
            .when(!includeProguardLocationReferences)
            .thenAdd("--no-proguard-location-reference")
            .when(conditionalKeepRules)
            .thenAdd("--proguard-conditional-keep-rules")
            .add("-o", linked)
            .execute(String.format("Linking %s", compiled.getManifest())));
    profiler.recordEndOf("fulllink");
    return ProtoApk.readFrom(optimize(compiled, linked));
  }

  /** Modes for overriding compression of a given file. */
  private enum CompressionOverride {
    DONT_CARE,
    FORCE_DEFLATED,
    FORCE_STORED
  }

  /*
   * Determine whether to override compression of given {@link DirectoryEntry}.
   */
  private CompressionOverride overrideCompression(DirectoryEntry entry) {
    String filename = entry.getFilename();
    if (filename.startsWith("assets/") && filename.endsWith(".apk")) {
      // This is solely to preserve legacy behavior, which could not otherwise be replicated with
      // command line flags - nested APKs are compressed in res/raw unless in
      // uncompressedExtensions, but are *never* compressed in assets.
      return CompressionOverride.FORCE_STORED;
    }
    if (filename.startsWith("res/")
        && filename.endsWith(".xml")
        && !uncompressedExtensions.contains(".xml")) {
      // b/186226111 - aapt2 optimize is overeager about declaring proto XML files incompressible
      // before conversion to binary, when their compression ratio generally gets much better.
      return CompressionOverride.FORCE_DEFLATED;
    }
    return CompressionOverride.DONT_CARE;
  }

  /** Retrieve a {@link Deflater} suitable for working on raw entry data (without headers). */
  private static Deflater getDeflater() {
    return new Deflater(Deflater.DEFAULT_COMPRESSION, /* nowrap= */ true);
  }
  /** Retrieve a {@link Inflater} suitable for working on raw entry data (without headers). */
  private static Inflater getInflater() {
    return new Inflater(/* nowrap= */ true);
  }

  /** Fix compression in {@code apk} according to {@link overrideCompression(DirectoryEntry)}. */
  private Path copyAndFixCompression(Path apk, Path workingDirectory) throws IOException {
    profiler.startTask("fixcompression");
    final Path outApk = workingDirectory.resolve("recompressed.apk");
    try (FileChannel inChannel = FileChannel.open(apk, StandardOpenOption.READ);
        FileChannel outChannel =
            FileChannel.open(outApk, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE); ) {
      final ZipIn zipIn = new ZipIn(inChannel, apk.toString());
      final ZipOut zipOut = new ZipOut(outChannel, outApk.toString());

      final EntryHandler entryHandler =
          (in, header, dirEntry, data) -> {
            final String filename = dirEntry.getFilename();

            short how = dirEntry.get(CENHOW);
            int siz = dirEntry.get(CENSIZ);
            switch (overrideCompression(dirEntry)) {
              case FORCE_DEFLATED:
                if (how == STORED) {
                  try (ByteArrayOutputStream byteStream = new ByteArrayOutputStream()) {
                    try (DeflaterOutputStream deflaterOutputStream =
                            new DeflaterOutputStream(byteStream, getDeflater());
                        WritableByteChannel channel = Channels.newChannel(deflaterOutputStream)) {
                      channel.write(data);
                    }
                    byte[] rawData = byteStream.toByteArray();
                    how = DEFLATED;
                    siz = rawData.length;
                    data = ByteBuffer.wrap(rawData);
                  }
                }
                break;
              case FORCE_STORED:
                if (how == DEFLATED) {
                  byte[] rawData = new byte[data.remaining()];
                  data.get(rawData);
                  try (InputStream byteStream = ByteSource.wrap(rawData).openStream();
                      InflaterInputStream inflaterInputStream =
                          new InflaterInputStream(byteStream, getInflater())) {
                    how = STORED;
                    siz = dirEntry.get(CENLEN);
                    data = ByteBuffer.wrap(ByteStreams.toByteArray(inflaterInputStream));
                  }
                }
                break;
              case DONT_CARE:
                break;
            }

            String comment = dirEntry.getComment();
            byte[] extra = dirEntry.getExtraData();
            zipOut.nextEntry(
                dirEntry
                    .clone(filename, extra, comment)
                    .set(CENHOW, how)
                    .set(CENSIZ, siz)
                    .set(CENTIM, DosTime.EPOCHISH.time));
            zipOut.write(
                header
                    .clone(filename, extra)
                    .set(LOCHOW, how)
                    .set(LOCSIZ, siz)
                    .set(LOCTIM, DosTime.EPOCHISH.time));
            zipOut.write(data);
            if ((header.get(LOCFLG) & LocalFileHeader.SIZE_MASKED_FLAG) != 0) {
              DataDescriptor desc =
                  DataDescriptor.allocate()
                      .set(EXTCRC, dirEntry.get(CENCRC))
                      .set(EXTSIZ, siz)
                      .set(EXTLEN, dirEntry.get(CENLEN));
              zipOut.write(desc);
            }
          };
      zipIn.scanEntries(entryHandler);
      zipOut.close();
      return outApk;
    } finally {
      profiler.recordEndOf("fixcompression");
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

  private Path optimize(CompiledResources compiled, Path protoApk) throws IOException {
    if (densities.size() < 2) {
      return protoApk;
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
            .add(protoApk.toString())
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
            outputAsProto ? protoApk.asApkPath() : convertProtoApkToBinary(protoApk),
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

  /** Convert a proto apk to binary. */
  public Path convertProtoApkToBinary(ProtoApk protoApk) {
    try {
      final Path protoApkPath = protoApk.asApkPath();
      final Path working =
          workingDirectory
              .resolve("link-proto")
              .resolve(replaceExtension(protoApkPath.getFileName().toString(), "working"));
      Files.createDirectories(working);
      final Path apk = working.resolve("binary.apk");
      logger.fine(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.DEFAULT)
              .add("convert")
              .when(Objects.equals(logger.getLevel(), Level.FINE))
              .thenAdd("-v")
              .add("-o", apk.toString())
              .add(protoApk.asApkPath().toString())
              .execute(String.format("Converting %s", protoApkPath)));
      return copyAndFixCompression(apk, working);
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

  public ResourceLinker includeProguardLocationReferences(
      boolean includeProguardLocationReferences) {
    this.includeProguardLocationReferences = includeProguardLocationReferences;
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
        .add("includeProguardLocationReferences", includeProguardLocationReferences)
        .toString();
  }
}
