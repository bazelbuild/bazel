// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import com.android.ide.common.res2.MergingException;

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import javax.annotation.concurrent.Immutable;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;

/**
 * Represents a collection of Android Resources.
 *
 * The AndroidDataSet is the primary building block for merging several AndroidDependencies
 * together. It extracts the android resource symbols (e.g. R.string.Foo) from the xml files to
 * allow an AndroidDataMerger to consume and produce a merged set of data.
 */
@Immutable
public class AndroidDataSet {
  /**
   * An AndroidDataPathWalker that collects DataAsset and DataResources for an AndroidDataSet.
   */
  private static final class AndroidDataSetBuildingPathWalker implements AndroidDataPathWalker {
    private final Set<MergeConflict> conflicts;
    private final Map<DataKey, DataAsset> assets;
    private final ResourceFileVisitor resourceVisitor;
    private static final ImmutableSet<FileVisitOption> FOLLOW_LINKS =
        ImmutableSet.of(FileVisitOption.FOLLOW_LINKS);
    private final Map<DataKey, DataResource> overwritingResources;
    private final Map<DataKey, DataResource> nonOverwritingResources;

    private static AndroidDataSetBuildingPathWalker create() {
      Map<DataKey, DataResource> overwritingResources = new HashMap<>();
      Map<DataKey, DataResource> nonOverwritingResources = new HashMap<>();
      final Map<DataKey, DataAsset> assets = new HashMap<>();
      Set<MergeConflict> conflicts = new HashSet<>();
      final ResourceFileVisitor resourceVisitor =
          new ResourceFileVisitor(conflicts, overwritingResources, nonOverwritingResources);
      return new AndroidDataSetBuildingPathWalker(
          conflicts, assets, overwritingResources, nonOverwritingResources, resourceVisitor);
    }

    private AndroidDataSetBuildingPathWalker(
        Set<MergeConflict> conflicts,
        Map<DataKey, DataAsset> assets,
        Map<DataKey, DataResource> overwritingResources,
        Map<DataKey, DataResource> nonOverwritingResources,
        ResourceFileVisitor resourceVisitor) {
      this.conflicts = conflicts;
      this.assets = assets;
      this.overwritingResources = overwritingResources;
      this.nonOverwritingResources = nonOverwritingResources;
      this.resourceVisitor = resourceVisitor;
    }

    @Override
    public void walkResources(Path path) throws IOException {

      Files.walkFileTree(path, FOLLOW_LINKS, Integer.MAX_VALUE, resourceVisitor);
    }

    @Override
    public void walkAssets(Path path) throws IOException {
      Files.walkFileTree(
          path,
          FOLLOW_LINKS,
          Integer.MAX_VALUE,
          new AssetFileVisitor(RelativeAssetPath.Factory.of(path), assets, conflicts));
    }

    /**
     * Creates an AndroidDataSet from the collected DataAsset and DataResource instances.
     */
    public AndroidDataSet createAndroidDataSet() throws MergingException {
      resourceVisitor.checkForErrors();
      return AndroidDataSet.of(
          ImmutableSet.copyOf(conflicts),
          ImmutableMap.copyOf(overwritingResources),
          ImmutableMap.copyOf(nonOverwritingResources),
          ImmutableMap.copyOf(assets));
    }
  }

  /**
   * A FileVisitor that walks the asset tree and extracts FileDataResources.
   */
  private static class AssetFileVisitor extends SimpleFileVisitor<Path> {
    private final RelativeAssetPath.Factory dataKeyFactory;
    private final Map<DataKey, DataAsset> assets;
    private final Set<MergeConflict> conflicts;

    public AssetFileVisitor(
        RelativeAssetPath.Factory dataKeyFactory,
        Map<DataKey, DataAsset> assets,
        Set<MergeConflict> conflicts) {
      this.dataKeyFactory = dataKeyFactory;
      this.assets = assets;
      this.conflicts = conflicts;
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
      if (!Files.isDirectory(path)) {
        RelativeAssetPath key = dataKeyFactory.create(path);
        FileDataResource asset = FileDataResource.of(key, path);
        if (assets.containsKey(key)) {
          conflicts.add(MergeConflict.between(key, assets.get(key), asset));
        } else {
          assets.put(key, asset);
        }
      }
      return super.visitFile(path, attrs);
    }
  }

  /**
   * A FileVisitor that walks a resource tree and extract FullyQualifiedName and resource values.
   */
  private static class ResourceFileVisitor extends SimpleFileVisitor<Path> {
    private final Set<MergeConflict> conflicts;
    private final Map<DataKey, DataResource> overwritingResources;
    private final Map<DataKey, DataResource> nonOverwritingResources;
    private final List<Exception> errors = new ArrayList<>();
    private boolean inValuesSubtree;
    private FullyQualifiedName.Factory fqnFactory;
    private final XMLInputFactory xmlInputFactory = XMLInputFactory.newFactory();

    public ResourceFileVisitor(
        Set<MergeConflict> conflicts,
        Map<DataKey, DataResource> overwritingResources,
        Map<DataKey, DataResource> nonOverwritingResources) {
      this.conflicts = conflicts;
      this.overwritingResources = overwritingResources;
      this.nonOverwritingResources = nonOverwritingResources;
    }

    private void checkForErrors() throws MergingException {
      if (!getErrors().isEmpty()) {
        StringBuilder errors = new StringBuilder();
        for (Exception e : getErrors()) {
          errors.append("\n").append(e.getMessage());
        }
        throw new MergingException(errors.toString());
      }
    }

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
        throws IOException {
      final String[] dirNameAndQualifiers = dir.getFileName().toString().split("-");
      inValuesSubtree = "values".equals(dirNameAndQualifiers[0]);
      fqnFactory = FullyQualifiedName.Factory.from(getQualifiers(dirNameAndQualifiers));
      return FileVisitResult.CONTINUE;
    }

    private List<String> getQualifiers(String[] dirNameAndQualifiers) {
      if (dirNameAndQualifiers.length == 1) {
        return ImmutableList.of();
      }
      return Arrays.asList(
          Arrays.copyOfRange(dirNameAndQualifiers, 1, dirNameAndQualifiers.length));
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
      try {
        if (!Files.isDirectory(path)) {
          if (inValuesSubtree) {
            List<DataResource> targetOverwritableResources = new ArrayList<>();
            List<DataResource> targetNonOverwritableResources = new ArrayList<>();
            XmlDataResource.fromPath(
                xmlInputFactory,
                path,
                fqnFactory,
                targetOverwritableResources,
                targetNonOverwritableResources);
            for (DataResource dataResource : targetOverwritableResources) {
              putOverwritingResource(dataResource);
            }
            for (DataResource dataResource : targetNonOverwritableResources) {
              if (!nonOverwritingResources.containsKey(dataResource.dataKey())) {
                nonOverwritingResources.put(dataResource.dataKey(), dataResource);
              }
            }
          } else {
            DataResource dataResource = FileDataResource.fromPath(path, fqnFactory);
            putOverwritingResource(dataResource);
          }
        }
      } catch (IllegalArgumentException | XMLStreamException e) {
        errors.add(e);
      }
      return super.visitFile(path, attrs);
    }

    private void putOverwritingResource(DataResource dataResource) {
      if (overwritingResources.containsKey(dataResource.dataKey())) {
        conflicts.add(
            MergeConflict.between(
                dataResource.dataKey(),
                dataResource,
                overwritingResources.get(dataResource.dataKey())));
      } else {
        overwritingResources.put(dataResource.dataKey(), dataResource);
      }
    }

    public List<Exception> getErrors() {
      return errors;
    }
  }

  /** Creates AndroidDataSet of conflicts, assets overwriting and nonOverwriting resources. */
  public static AndroidDataSet of(
      ImmutableSet<MergeConflict> conflicts,
      ImmutableMap<DataKey, DataResource> overwritingResources,
      ImmutableMap<DataKey, DataResource> nonOverwritingResources,
      ImmutableMap<DataKey, DataAsset> assets) {
    return new AndroidDataSet(conflicts, overwritingResources, nonOverwritingResources, assets);
  }

  /**
   * Creates an AndroidDataSet from an UnvalidatedAndroidData.
   *
   * The adding process parses out all the provided symbol into DataResources and DataAssets
   * objects.
   *
   * @param primary The primary data to parse into DataResources and DataAssets.
   * @throws IOException when there are issues with reading files.
   * @throws MergingException when there is invalid resource information.
   */
  public static AndroidDataSet from(UnvalidatedAndroidData primary)
      throws IOException, MergingException {
    final AndroidDataSetBuildingPathWalker pathWalker = AndroidDataSetBuildingPathWalker.create();
    primary.walk(pathWalker);
    return pathWalker.createAndroidDataSet();
  }

  /**
   * Creates an AndroidDataSet from a list of DependencyAndroidData instances.
   *
   * The adding process parses out all the provided symbol into DataResources and DataAssets
   * objects.
   *
   * @param dependencyAndroidDataList The dependency data to parse into DataResources and
   *        DataAssets.
   * @throws IOException when there are issues with reading files.
   * @throws MergingException when there is invalid resource information.
   */
  public static AndroidDataSet from(List<DependencyAndroidData> dependencyAndroidDataList)
      throws IOException, MergingException {
    final AndroidDataSetBuildingPathWalker pathWalker = AndroidDataSetBuildingPathWalker.create();
    for (DependencyAndroidData data : dependencyAndroidDataList) {
      data.walk(pathWalker);
    }
    return pathWalker.createAndroidDataSet();
  }

  private final ImmutableSet<MergeConflict> conflicts;
  private final ImmutableMap<DataKey, DataResource> overwritingResources;
  private final ImmutableMap<DataKey, DataResource> nonOverwritingResources;
  private final ImmutableMap<DataKey, DataAsset> assets;

  private AndroidDataSet(
      ImmutableSet<MergeConflict> conflicts,
      ImmutableMap<DataKey, DataResource> overwritingResources,
      ImmutableMap<DataKey, DataResource> nonOverwritingResources,
      ImmutableMap<DataKey, DataAsset> assets) {
    this.conflicts = conflicts;
    this.overwritingResources = overwritingResources;
    this.nonOverwritingResources = nonOverwritingResources;
    this.assets = assets;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("overwritingResources", overwritingResources)
        .add("nonOverwritingResources", nonOverwritingResources)
        .add("assets", assets)
        .toString();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof AndroidDataSet)) {
      return false;
    }
    AndroidDataSet that = (AndroidDataSet) other;
    return Objects.equals(overwritingResources, that.overwritingResources)
        && Objects.equals(nonOverwritingResources, that.nonOverwritingResources)
        && Objects.equals(conflicts, that.conflicts)
        && Objects.equals(assets, that.assets);
  }

  @Override
  public int hashCode() {
    return Objects.hash(conflicts, overwritingResources, nonOverwritingResources, assets);
  }

  /**
   * Returns a list of resources that would overwrite other values when defined.
   *
   * <p>Example:
   *
   * A string resource (string.Foo=bar) could be redefined at string.Foo=baz.
   *
   * @return A map of key -&gt; overwriting resources.
   */
  @VisibleForTesting
  public Map<DataKey, DataResource> getOverwritingResources() {
    return overwritingResources;
  }

  /**
   * Returns a list of resources that would not overwrite other values when defined.
   *
   * <p>Example:
   *
   * A id resource (id.Foo) could be redefined at id.Foo with no adverse effects.
   *
   * @return A map of key -&gt; non-overwriting resources.
   */
  @VisibleForTesting
  public Map<DataKey, DataResource> getNonOverwritingResources() {
    return nonOverwritingResources;
  }

  /**
   * Returns a list of assets.
   *
   * Assets always overwrite during merging, just like overwriting resources.
   * <p>Example:
   *
   * A text asset (foo/bar.txt, containing fooza) could be replaced with (foo/bar.txt, containing
   * ouza!) depending on the merging process.
   *
   * @return A map of key -&gt; assets.
   */
  public Map<DataKey, DataAsset> getAssets() {
    return assets;
  }

  boolean containsOverwritable(DataKey name) {
    return overwritingResources.containsKey(name);
  }

  Iterable<Map.Entry<DataKey, DataResource>> iterateOverwritableEntries() {
    return overwritingResources.entrySet();
  }

  boolean containsAsset(DataKey name) {
    return assets.containsKey(name);
  }

  Iterable<Map.Entry<DataKey, DataAsset>> iterateAssetEntries() {
    return assets.entrySet();
  }

  MergeConflict foundResourceConflict(DataKey key, DataResource value) {
    return MergeConflict.between(key, overwritingResources.get(key), value);
  }

  MergeConflict foundAssetConflict(DataKey key, DataAsset value) {
    return MergeConflict.between(key, assets.get(key), value);
  }

  ImmutableMap<DataKey, DataResource> mergeNonOverwritable(AndroidDataSet other) {
    Map<DataKey, DataResource> merged = new HashMap<>(other.nonOverwritingResources);
    merged.putAll(nonOverwritingResources);
    return ImmutableMap.copyOf(merged);
  }

  ImmutableSet<MergeConflict> conflicts() {
    return conflicts;
  }
}
