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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import com.android.ide.common.res2.AssetSet;
import com.android.ide.common.res2.ResourceSet;

import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Android data that has yet to be merged and validated, the primary data for the Processor.
 *
 * <p>The life cycle of AndroidData goes:
 * {@link UnvalidatedAndroidData} -> {@link MergedAndroidData} -> {@link DensityFilteredAndroidData}
 *      -> {@link DependencyAndroidData}
 */
class UnvalidatedAndroidData {
  static final Pattern VALID_REGEX = Pattern.compile(".*:.*:.+");

  public static UnvalidatedAndroidData valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  @VisibleForTesting
  static UnvalidatedAndroidData valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(
          text + " is not in the format 'resources[#resources]:assets[#assets]:manifest'");
    }
    String[] parts = text.split(":");
    return new UnvalidatedAndroidData(
        splitPaths(parts[0], fileSystem),
        splitPaths(parts[1], fileSystem),
        exists(fileSystem.getPath(parts[2])));
  }

  private static ImmutableList<Path> splitPaths(String pathsString, FileSystem fileSystem) {
    if (pathsString.length() == 0) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Path> paths = new ImmutableList.Builder<>();
    for (String pathString : pathsString.split("#")) {
      paths.add(exists(fileSystem.getPath(pathString)));
    }
    return paths.build();
  }

  private static Path exists(Path path) {
    if (!Files.exists(path)) {
      throw new IllegalArgumentException(path + " does not exist");
    }
    return path;
  }

  private final Path manifest;
  private final ImmutableList<Path> assetDirs;
  private final ImmutableList<Path> resourceDirs;

  public UnvalidatedAndroidData(
      ImmutableList<Path> resourceDirs, ImmutableList<Path> assetDirs, Path manifest) {
    this.resourceDirs = resourceDirs;
    this.assetDirs = assetDirs;
    this.manifest = manifest;
  }

  public Path getManifest() {
    return manifest;
  }

  public AssetSet addToAssets(AssetSet assets) {
    for (Path assetDir : assetDirs) {
      assets.addSource(assetDir.toFile());
    }
    return assets;
  }

  public ResourceSet addToResourceSet(ResourceSet resources) {
    for (Path resourceDir : resourceDirs) {
      resources.addSource(resourceDir.toFile());
    }
    return resources;
  }

  public UnvalidatedAndroidData modify(ImmutableList<DirectoryModifier> modifiers) {
    ImmutableList<Path> modifiedResources = resourceDirs;
    ImmutableList<Path> modifiedAssets = assetDirs;
    for (DirectoryModifier modifier : modifiers) {
      modifiedAssets = modifier.modify(modifiedAssets);
      modifiedResources = modifier.modify(modifiedResources);
    }
    return new UnvalidatedAndroidData(modifiedResources, modifiedAssets, manifest);
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

  /**
   * Adds all the resource directories as ResourceSets. This acts a loose merge
   * strategy as it does not test for overrides.
   * @param resourceSets A list of resource sets to append to.
   */
  void addAsResourceSets(List<ResourceSet> resourceSets) {
    for (Path resourceDir : resourceDirs) {
      ResourceSet set = new ResourceSet("primary:" + resourceDir.toString());
      set.addSource(resourceDir.toFile());
      resourceSets.add(set);
    }
  }

  /**
   * Adds all the asset directories as AssetSets. This acts a loose merge
   * strategy as it does not test for overrides.
   * @param assetSets A list of asset sets to append to.
   */
  void addAsAssetSets(List<AssetSet> assetSets) {
    for (Path assetDir : assetDirs) {
      AssetSet set = new AssetSet("primary:" + assetDir.toString());
      set.addSource(assetDir.toFile());
      assetSets.add(set);
    }
  }

  public void walkResources(final FileVisitor<Path> fileVisitor) throws IOException {
    for (Path path : resourceDirs) {
      Files.walkFileTree(
          path, ImmutableSet.of(FileVisitOption.FOLLOW_LINKS), Integer.MAX_VALUE, fileVisitor);
    }
  }
}
