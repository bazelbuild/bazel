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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.ManifestContainer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Contains reference to the aapt2 generated .flat file archive and a manifest.
 *
 * <p>This represents the state between the aapt2 compile and link actions.
 */
public class CompiledResources implements ManifestContainer {

  /**
   * File extension for persisting {@code <resources tool:* />} attributes. These files are bundled
   * with *.flat files in the {@link resources} ZIP file and ignored by aapt2. The format is
   * internal to ResourceProcessorBusyBox and can be changed at any time.
   */
  // TODO(b/143382754): aapt2 should be serializing these directly.
  public static final String ATTRIBUTES_FILE_EXTENSION = ".attributes";

  private final Path resources;
  @Nullable private final Path manifest;
  private final List<Path> assetsDirs;
  @Nullable private final Path stableIds;

  /**
   * @param resources The path to the zip file containing the compiled resource result of aapt2
   * @param manifest The path to the manifest file for this set of resources
   * @param assetsDirs The list of asset directors for the target of this set of resources
   * @param stableIds The path to the file containing resource ID's to keep stable
   */
  private CompiledResources(
      Path resources, @Nullable Path manifest, List<Path> assetsDirs, @Nullable Path stableIds) {
    this.resources = resources;
    this.manifest = manifest;
    this.assetsDirs = assetsDirs;
    this.stableIds = stableIds;
  }

  public static CompiledResources from(Path resources) {
    return from(resources, null, ImmutableList.of());
  }

  public static CompiledResources from(Path resources, Path manifest) {
    return from(resources, manifest, ImmutableList.of());
  }

  public static CompiledResources from(
      Path resources, @Nullable Path manifest, @Nullable List<Path> assetDirs) {
    return new CompiledResources(
        resources, manifest, assetDirs != null ? assetDirs : ImmutableList.of(), null);
  }

  /** This zip file contains resource flat files that are the result of aapt2 compile */
  public Path getZip() {
    return resources;
  }

  /** Copies resources archive to a path and returns the new {@link CompiledResources} */
  public CompiledResources copyResourcesZipTo(Path destination) throws IOException {
    return new CompiledResources(
        Files.copy(resources, destination), manifest, assetsDirs, stableIds);
  }

  @Override
  @Nullable
  public Path getManifest() {
    if (manifest != null) {
      Preconditions.checkState(Files.exists(manifest));
    }
    return manifest;
  }

  public List<String> getAssetsStrings() {
    return assetsDirs.stream()
        .peek(
            p -> {
              Preconditions.checkArgument(Files.exists(p), "does not exist %s", p);
            })
        .map(Path::toString)
        .collect(Collectors.toList());
  }

  public CompiledResources processManifest(Function<Path, Path> processManifest) {
    return new CompiledResources(resources, processManifest.apply(manifest), assetsDirs, stableIds);
  }

  public CompiledResources addStableIds(Path stableIds) {
    return new CompiledResources(resources, manifest, assetsDirs, stableIds);
  }

  public Optional<Path> getStableIds() {
    return Optional.ofNullable(stableIds);
  }
}
