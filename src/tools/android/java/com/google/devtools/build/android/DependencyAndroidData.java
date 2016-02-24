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
import com.google.common.collect.ImmutableSet;

import com.android.builder.dependency.SymbolFileProvider;
import com.android.ide.common.res2.AssetSet;
import com.android.ide.common.res2.ResourceSet;

import java.io.File;
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
 * Contains the assets, resources, manifest and resource symbols for an android_library dependency.
 *
 * <p>
 * This class serves the role of both a processed MergedAndroidData and a dependency exported from
 * another invocation of the AndroidResourcesProcessorAction. Since it's presumed to be cheaper to
 * only pass the derived artifact (rTxt) rather that the entirety of the processed dependencies (png
 * crunching and resource processing should be saved for the final AndroidResourcesProcessorAction
 * invocation) AndroidData can have multiple roots for resources and assets.
 * </p>
 */
class DependencyAndroidData {
  static final Pattern VALID_REGEX = Pattern.compile(".*:.*:.+:.+(:.*)?");

  public static DependencyAndroidData valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  @VisibleForTesting
  static DependencyAndroidData valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(
          text
              + " is not in the format 'resources[#resources]:assets[#assets]:manifest:"
              + "r.txt:symbols.txt'");
    }
    String[] parts = text.split("\\:");
    // TODO(bazel-team): Handle the local-r.txt file.
    // The local R is optional -- if it is missing, we'll use the full R.txt
    return new DependencyAndroidData(
        splitPaths(parts[0], fileSystem),
        parts[1].length() == 0 ? ImmutableList.<Path>of() : splitPaths(parts[1], fileSystem),
        exists(fileSystem.getPath(parts[2])),
        exists(fileSystem.getPath(parts[3])),
        parts.length == 5 ? fileSystem.getPath(parts[4]) : null);
  }

  private static ImmutableList<Path> splitPaths(String pathsString, FileSystem fileSystem) {
    if (pathsString.trim().isEmpty()) {
      return ImmutableList.<Path>of();
    }
    ImmutableList.Builder<Path> paths = new ImmutableList.Builder<>();
    for (String pathString : pathsString.split("#")) {
      Preconditions.checkArgument(!pathString.trim().isEmpty());
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

  private final Path rTxt;
  private final Path manifest;
  private final ImmutableList<Path> assetDirs;
  private final ImmutableList<Path> resourceDirs;
  private final Path symbolsTxt;

  public DependencyAndroidData(
      ImmutableList<Path> resourceDirs,
      ImmutableList<Path> assetDirs,
      Path manifest,
      Path rTxt,
      Path symbolsTxt) {
    this.resourceDirs = resourceDirs;
    this.assetDirs = assetDirs;
    this.manifest = manifest;
    this.rTxt = rTxt;
    this.symbolsTxt = symbolsTxt;
  }

  public SymbolFileProvider asSymbolFileProvider() {
    return new SymbolFileProvider() {
      @Override
      public File getManifest() {
        return manifest.toFile();
      }

      @Override
      public File getSymbolFile() {
        return rTxt == null ? null : rTxt.toFile();
      }
    };
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

  /**
   * Adds all the resource directories as ResourceSets. This acts a loose merge
   * strategy as it does not test for overrides.
   * @param resourceSets A list of resource sets to append to.
   */
  void addAsResourceSets(List<ResourceSet> resourceSets) {
    for (Path resourceDir : resourceDirs) {
      ResourceSet set = new ResourceSet("dependency:" + resourceDir.toString());
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
      AssetSet set = new AssetSet("dependency:" + assetDir.toString());
      set.addSource(assetDir.toFile());
      assetSets.add(set);
    }
  }

  @Override
  public String toString() {
    return String.format(
        "AndroidData(%s, %s, %s, %s, %s)", resourceDirs, assetDirs, manifest, rTxt, symbolsTxt);
  }

  @Override
  public int hashCode() {
    return Objects.hash(resourceDirs, assetDirs, manifest, rTxt, symbolsTxt);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof DependencyAndroidData)) {
      return false;
    }
    DependencyAndroidData other = (DependencyAndroidData) obj;
    return Objects.equals(other.resourceDirs, resourceDirs)
        && Objects.equals(other.assetDirs, assetDirs)
        && Objects.equals(other.rTxt, rTxt)
        && Objects.equals(other.symbolsTxt, symbolsTxt)
        && Objects.equals(other.manifest, manifest);
  }

  public DependencyAndroidData modify(ImmutableList<DirectoryModifier> modifiers) {
    ImmutableList<Path> modifiedResources = resourceDirs;
    ImmutableList<Path> modifiedAssets = assetDirs;
    for (DirectoryModifier modifier : modifiers) {
      modifiedAssets = modifier.modify(modifiedAssets);
      modifiedResources = modifier.modify(modifiedResources);
    }
    return new DependencyAndroidData(modifiedResources, modifiedAssets, manifest, rTxt, null);
  }

  public void walk(final FileVisitor<Path> fileVisitor) throws IOException {
    for (Path path : resourceDirs) {
      Files.walkFileTree(
          path, ImmutableSet.of(FileVisitOption.FOLLOW_LINKS), Integer.MAX_VALUE, fileVisitor);
    }
  }
}
