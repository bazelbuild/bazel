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
package com.google.devtools.build.android;

import static com.google.common.base.Predicates.not;

import com.android.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.AndroidResourceOutputs.ZipBuilder;
import com.google.devtools.build.android.AndroidResourceOutputs.ZipBuilderVisitorWithDirectories;
import com.google.devtools.build.android.aapt2.ProtoApk;
import com.google.devtools.build.android.aapt2.ProtoResourceUsageAnalyzer;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.proto.SerializeFormat.ToolAttributes;
import com.google.protobuf.ExtensionRegistry;
import java.io.Closeable;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

/** Represents a collection of raw, merged resources with an optional id list. */
public class ResourcesZip {

  static final Logger logger = Logger.getLogger(ResourcesZip.class.toString());

  @Nullable private final Path resourcesRoot;
  @Nullable private final Path assetsRoot;
  @Nullable private final Path apkWithAssets;
  @Nullable private final Path proto;
  @Nullable private final Path attributes;
  @Nullable private final Path packages;
  @Nullable private final Path ids;

  private ResourcesZip(
      @Nullable Path resourcesRoot,
      @Nullable Path assetsRoot,
      @Nullable Path ids,
      @Nullable Path apkWithAssets,
      @Nullable Path proto,
      @Nullable Path attributes,
      @Nullable Path packages) {
    this.resourcesRoot = resourcesRoot;
    this.assetsRoot = assetsRoot;
    this.ids = ids;
    this.apkWithAssets = apkWithAssets;
    this.proto = proto;
    this.attributes = attributes;
    this.packages = packages;
  }

  /**
   * @param resourcesRoot The root of the raw resources.
   * @param assetsRoot The root of the raw assets.
   */
  public static ResourcesZip from(Path resourcesRoot, Path assetsRoot) {
    return new ResourcesZip(resourcesRoot, assetsRoot, null, null, null, null, null);
  }

  /**
   * @param proto apk in proto format.
   * @param attributes Tooling attributes.
   * @param resourceIds Optional path to a file containing the resource ids.
   * @param packages Optional path to a file containing the dependency packages.
   */
  public static ResourcesZip fromApkWithProto(
      Path proto, Path attributes, Path resourceIds, Path packages) {
    return new ResourcesZip(
        /* resourcesRoot= */ null,
        /* assetsRoot= */ null,
        resourceIds != null && Files.exists(resourceIds) ? resourceIds : null,
        /* apkWithAssets= */ null,
        proto,
        attributes,
        packages);
  }

  /** Creates a ResourcesZip from an archive by expanding into the workingDirectory. */
  public static ResourcesZip createFrom(Path resourcesZip, Path workingDirectory)
      throws IOException {
    // Expand resource files zip into working directory.
    final ZipFile zipFile = new ZipFile(resourcesZip.toFile());

    zipFile.stream()
        .filter(not(ZipEntry::isDirectory))
        .forEach(
            entry -> {
              Path output = workingDirectory.resolve(entry.getName());
              try {
                Files.createDirectories(output.getParent());
                try (FileOutputStream fos = new FileOutputStream(output.toFile())) {
                  ByteStreams.copy(zipFile.getInputStream(entry), fos);
                }
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
            });
    return new ResourcesZip(
        Files.createDirectories(workingDirectory.resolve("res")),
        Files.createDirectories(workingDirectory.resolve("assets")),
        ifExists(workingDirectory.resolve("ids.txt")),
        /** apkWithAssets */
        null,
        ifExists(workingDirectory.resolve("apk.pb")),
        ifExists(workingDirectory.resolve("tools.attributes.pb")),
        ifExists(workingDirectory.resolve("packages.txt")));
  }

  private static Path ifExists(Path path) {
    return Files.exists(path) ? path : null;
  }

  /**
   * Creates a zip file containing the provided android resources and assets.
   *
   * @param output The path to write the zip file
   * @param compress Whether or not to compress the content
   * @throws IOException
   */
  public void writeTo(Path output, boolean compress) throws IOException {
    try (final ZipBuilder zip = ZipBuilder.createFor(output)) {
      if (resourcesRoot != null && Files.exists(resourcesRoot)) {
        ZipBuilderVisitorWithDirectories visitor =
            new ZipBuilderVisitorWithDirectories(zip, resourcesRoot, "res");
        visitor.setCompress(compress);
        Files.walkFileTree(resourcesRoot, visitor);
        if (!Files.exists(resourcesRoot.resolve("values/public.xml"))) {
          // add an empty public xml, if one doesn't exist. The ResourceUsageAnalyzer expects one.
          visitor.addEntry(
              resourcesRoot.resolve("values").resolve("public.xml"),
              "<resources></resources>".getBytes(StandardCharsets.UTF_8));
        }
        visitor.writeEntries();
      }

      if (apkWithAssets != null && Files.exists(apkWithAssets)) {
        ZipFile apkZip = new ZipFile(apkWithAssets.toString());
        if (apkZip.getEntry("assets/") == null) {
          zip.addEntry("assets/", new byte[0], compress ? ZipEntry.DEFLATED : ZipEntry.STORED);
        }
        apkZip.stream()
            .filter(entry -> entry.getName().startsWith("assets/"))
            .forEach(
                entry -> {
                  try {
                    zip.addEntry(entry, ByteStreams.toByteArray(apkZip.getInputStream(entry)));
                  } catch (IOException e) {
                    throw new RuntimeException(e);
                  }
                });
      } else if (assetsRoot != null && Files.exists(assetsRoot)) {
        ZipBuilderVisitorWithDirectories visitor =
            new ZipBuilderVisitorWithDirectories(zip, assetsRoot, "assets");
        visitor.setCompress(compress);
        Files.walkFileTree(assetsRoot, visitor);
        visitor.writeEntries();
      }
      try {
        if (ids != null) {
          zip.addEntry("ids.txt", Files.readAllBytes(ids), ZipEntry.STORED);
        }

        if (proto != null && Files.exists(proto)) {
          zip.addEntry("apk.pb", Files.readAllBytes(proto), ZipEntry.STORED);
        }

        if (attributes != null && Files.exists(attributes)) {
          zip.addEntry("tools.attributes.pb", Files.readAllBytes(attributes), ZipEntry.STORED);
        }

        if (packages != null && Files.exists(packages)) {
          zip.addEntry("packages.txt", Files.readAllBytes(packages), ZipEntry.STORED);
        }

      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /**
   * Shrinks the apk using a protocol buffer apk.
   *
   * @param packages The packages of the dependencies. Used to analyze the java code for resource
   *     references.
   * @param classJar Used to find resource references in java.
   * @param rTxt R.txt file listing all resources.
   * @param proguardMapping Mapping used to decode java references.
   * @param logFile Destination of the resource shrinker log.
   * @param workingDirectory Temporary directory for intermediate artifacts.
   * @return A ShrunkProtoApk, which must be closed when finished.
   * @throws ParserConfigurationException thrown when the xml parsing not possible.
   * @throws IOException thrown when the filesystem is going pear shaped.
   * @throws SAXException thrown when the xml parsing goes badly.
   */
  @CheckReturnValue
  public ShrunkProtoApk shrinkUsingProto(
      Set<String> packages,
      Path classJar,
      Path rTxt,
      Path proguardMapping,
      Path logFile,
      Path workingDirectory)
      throws ParserConfigurationException, IOException {
    Path shrunkApkProto = workingDirectory.resolve("shrunk." + ResourceLinker.PROTO_EXTENSION);
    Path resourcesConfig = workingDirectory.resolve("resources.cfg");

    try (final ProtoApk apk = ProtoApk.readFrom(proto)) {
      // record resources and manifest
      final ProtoResourceUsageAnalyzer analyzer =
          new ProtoResourceUsageAnalyzer(packages, rTxt, proguardMapping, resourcesConfig, logFile);

      ProtoApk shrink = analyzer.shrink(apk, classJar, shrunkApkProto, parseToolAttributes());
      return new ShrunkProtoApk(shrink, resourcesConfig, logFile);
    }
  }

  @VisibleForTesting
  ImmutableListMultimap<String, String> parseToolAttributes() throws IOException {
    return ToolAttributes.parseFrom(
            Files.readAllBytes(attributes), ExtensionRegistry.getEmptyRegistry())
        .getAttributesMap()
        .entrySet()
        .stream()
        .collect(
            ImmutableListMultimap.flatteningToImmutableListMultimap(
                Entry::getKey, e -> e.getValue().getValuesList().stream()));
  }

  public List<String> asPackages() throws IOException {
    return packages != null
        ? Files.readAllLines(packages, StandardCharsets.UTF_8)
        : ImmutableList.of();
  }

  Path asApk() {
    return proto;
  }

  static class ShrunkProtoApk implements Closeable {
    private final ProtoApk apk;
    private final Path resourcesConfig;
    private final Path report;

    ShrunkProtoApk(ProtoApk apk, Path resourcesConfig, Path report) {
      this.apk = apk;
      this.resourcesConfig = resourcesConfig;
      this.report = report;
    }

    ShrunkProtoApk writeBinaryTo(ResourceLinker linker, Path binaryOut, boolean writeAsProto)
        throws IOException {
      Files.copy(
          writeAsProto ? apk.asApkPath() : linker.convertProtoApkToBinary(apk),
          binaryOut,
          StandardCopyOption.REPLACE_EXISTING);
      return this;
    }

    void writeResourcesConfigTo(Path resourcesConfigOut) throws Exception {
      Files.copy(resourcesConfig, resourcesConfigOut);
    }

    ShrunkProtoApk writeReportTo(Path reportOut) throws IOException {
      Files.copy(report, reportOut);
      return this;
    }

    ShrunkProtoApk writeResourcesToZip(Path resourcesZip) throws IOException {
      try (final ZipBuilder zip = ZipBuilder.createFor(resourcesZip)) {
        zip.addEntry("apk.pb", Files.readAllBytes(apk.asApkPath()), ZipEntry.STORED);
      }
      return this;
    }

    @Override
    public void close() throws IOException {
      apk.close();
    }
  }
}
