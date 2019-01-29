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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Android data that has yet to be merged and validated, the primary data for the Processor.
 *
 * <p>The life cycle of AndroidData goes: {@link UnvalidatedAndroidData} -> {@link
 * MergedAndroidData} -> {@link DensityFilteredAndroidData} -> {@link DependencyAndroidData}
 */
class UnvalidatedAndroidData extends UnvalidatedAndroidDirectories {

  private static final Pattern VALID_REGEX = Pattern.compile(".*:.*:.+");

  public static final String EXPECTED_FORMAT = "resources[#resources]:assets[#assets]:manifest";

  public static UnvalidatedAndroidData valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  @VisibleForTesting
  static UnvalidatedAndroidData valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(text + " is not in the format '" + EXPECTED_FORMAT + "'");
    }
    String[] parts = text.split(":");
    return new UnvalidatedAndroidData(
        splitPaths(parts[0], fileSystem),
        splitPaths(parts[1], fileSystem),
        exists(fileSystem.getPath(parts[2])));
  }

  private final Path manifest;

  public UnvalidatedAndroidData(
      ImmutableList<Path> resourceDirs, ImmutableList<Path> assetDirs, Path manifest) {
    super(resourceDirs, assetDirs);
    this.manifest = manifest;
  }

  public Path getManifest() {
    return manifest;
  }

  @Override
  public String toString() {
    return String.format("UnvalidatedAndroidData(%s, %s, %s)", resourceDirs, assetDirs, manifest);
  }

  @Override
  public int hashCode() {
    return Objects.hash(resourceDirs, assetDirs, manifest);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof UnvalidatedAndroidData)) {
      return false;
    }
    UnvalidatedAndroidData other = (UnvalidatedAndroidData) obj;
    return Objects.equals(other.resourceDirs, resourceDirs)
        && Objects.equals(other.assetDirs, assetDirs)
        && Objects.equals(other.manifest, manifest);
  }

  public CompiledResources compile(ResourceCompiler compiler, Path workingDirectory)
      throws IOException, ExecutionException, InterruptedException {
    for (Path resourceDir : resourceDirs) {
      compiler.queueDirectoryForCompilation(resourceDir);
    }
    return archiveCompiledResources(
        compiler.getCompiledArtifacts(),
        workingDirectory,
        workingDirectory.resolve("compiled.zip"));
  }

  protected CompiledResources archiveCompiledResources(
      List<Path> resources, Path workingDirectory, Path output) throws IOException {
    return CompiledResources.from(
        AndroidResourceOutputs.archiveCompiledResources(
            output, workingDirectory, workingDirectory, resources),
        manifest,
        assetDirs);
  }

  /* Processes the resources for databinding annotations if dataBindingOut is defined. */
  public UnvalidatedAndroidData processDataBindings(
      @Nullable Path dataBindingOut, String packagePath, Path dataBindingWorkingDirectory)
      throws IOException {

    if (dataBindingOut == null) {
      return this;
    }

    Preconditions.checkNotNull(manifest);
    Preconditions.checkNotNull(packagePath);

    final List<Path> processed = new ArrayList<>();
    final Path metadataWorkingDirectory =
        Files.createDirectory(dataBindingWorkingDirectory.resolve("metadata"));
    final Path databindingResourceRoot = dataBindingWorkingDirectory.resolve("resources");
    for (Path resource : resourceDirs) {
      processed.add(
          AndroidResourceProcessor.processDataBindings(
              databindingResourceRoot,
              resource,
              metadataWorkingDirectory,
              packagePath,
              false));
    }

    AndroidResourceOutputs.archiveDirectory(metadataWorkingDirectory, dataBindingOut);

    return new UnvalidatedAndroidData(ImmutableList.copyOf(processed), assetDirs, manifest) {
      @Override
      protected CompiledResources archiveCompiledResources(
          List<Path> resources, Path workingDirectory, Path output) throws IOException {
        // Update the archiving to ensure that the resources are correctly placed.
        return CompiledResources.from(
            AndroidResourceOutputs.archiveCompiledResources(
                output, databindingResourceRoot, workingDirectory, resources),
            manifest,
            assetDirs);
      }
    };
  }
}
