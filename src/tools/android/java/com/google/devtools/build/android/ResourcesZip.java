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

import com.android.build.gradle.tasks.ResourceUsageAnalyzer;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.AndroidResourceOutputs.ZipBuilder;
import com.google.devtools.build.android.AndroidResourceOutputs.ZipBuilderVisitorWithDirectories;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

/** Represents a collection of raw, merged resources with an optional id list. */
public class ResourcesZip {

  private final Path resourcesRoot;
  private final Path assetsRoot;
  private final Optional<Path> ids;

  private ResourcesZip(Path resourcesRoot, Path assetsRoot, Optional<Path> ids) {
    this.resourcesRoot = resourcesRoot;
    this.assetsRoot = assetsRoot;
    this.ids = ids;
  }

  /**
   * @param resourcesRoot The root of the raw resources.
   * @param assetsRoot The root of the raw assets.
   */
  public static ResourcesZip from(Path resourcesRoot, Path assetsRoot) {
    return new ResourcesZip(resourcesRoot, assetsRoot, Optional.empty());
  }

  /**
   * @param resourcesRoot The root of the raw resources.
   * @param assetsRoot The root of the raw assets.
   * @param resourceIds Optional path to a file containing the resource ids.
   */
  public static ResourcesZip from(Path resourcesRoot, Path assetsRoot, Path resourceIds) {
    return new ResourcesZip(
        resourcesRoot, assetsRoot, Optional.of(resourceIds).filter(Files::exists));
  }

  /** Creates a ResourcesZip from an archive by expanding into the workingDirectory. */
  public static ResourcesZip createFrom(Path resourcesZip, Path workingDirectory)
      throws IOException {
    // Expand resource files zip into working directory.
    final ZipFile zipFile = new ZipFile(resourcesZip.toFile());

    zipFile
        .stream()
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
    return from(
        Files.createDirectories(workingDirectory.resolve("res")),
        Files.createDirectories(workingDirectory.resolve("assets")),
        workingDirectory.resolve("ids.txt"));
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
      if (Files.exists(resourcesRoot)) {
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
      if (Files.exists(assetsRoot)) {
        ZipBuilderVisitorWithDirectories visitor =
            new ZipBuilderVisitorWithDirectories(zip, assetsRoot, "assets");
        visitor.setCompress(compress);
        Files.walkFileTree(assetsRoot, visitor);
        visitor.writeEntries();
      }

      ids.ifPresent(
          p -> {
            try {
              zip.addEntry("ids.txt", Files.readAllBytes(p), ZipEntry.STORED);
            } catch (IOException e) {
              throw new RuntimeException(e);
            }
          });
    }
  }

  /** Removes unused resources from the archived resources. */
  public ShrunkResources shrink(
      Set<String> packages,
      Path rTxt,
      Path classJar,
      Path manifest,
      @Nullable Path proguardMapping,
      Path logFile,
      Path workingDirectory)
      throws ParserConfigurationException, IOException, SAXException {

    new ResourceUsageAnalyzer(
            packages, rTxt, classJar, manifest, proguardMapping, resourcesRoot, logFile)
        .shrink(workingDirectory);
    return ShrunkResources.of(
        new ResourcesZip(workingDirectory, assetsRoot, ids),
        new UnvalidatedAndroidData(
            ImmutableList.of(workingDirectory), ImmutableList.of(assetsRoot), manifest));
  }

  static class ShrunkResources {

    private ResourcesZip resourcesZip;
    private UnvalidatedAndroidData unvalidatedAndroidData;

    private ShrunkResources(
        ResourcesZip resourcesZip, UnvalidatedAndroidData unvalidatedAndroidData) {
      this.resourcesZip = resourcesZip;
      this.unvalidatedAndroidData = unvalidatedAndroidData;
    }

    public static ShrunkResources of(
        ResourcesZip resourcesZip, UnvalidatedAndroidData unvalidatedAndroidData) {
      return new ShrunkResources(resourcesZip, unvalidatedAndroidData);
    }

    public ShrunkResources writeArchiveTo(Path archivePath, boolean compress) throws IOException {
      resourcesZip.writeTo(archivePath, compress);
      return this;
    }

    public CompiledResources compile(ResourceCompiler compiler, Path workingDirectory)
        throws InterruptedException, ExecutionException, IOException {
      return unvalidatedAndroidData
          .compile(compiler, workingDirectory)
          .addStableIds(resourcesZip.ids);
    }
  }
}
