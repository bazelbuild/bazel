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
import com.google.common.collect.Iterables;

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
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;

import javax.annotation.concurrent.Immutable;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;

/**
 * Represents a collection of Android Resources.
 *
 * The ParsedAndroidData is the primary building block for merging several AndroidDependencies
 * together. It extracts the android resource symbols (e.g. R.string.Foo) from the xml files to
 * allow an AndroidDataMerger to consume and produce a merged set of data.
 */
@Immutable
public class ParsedAndroidData {
  /** A Consumer style interface that will appendTo a DataKey and DataValue. */
  interface KeyValueConsumer<K extends DataKey, V extends DataValue> {
    void consume(K key, V value);
  }

  @VisibleForTesting
  static class NonOverwritableConsumer implements KeyValueConsumer<DataKey, DataResource> {

    private Map<DataKey, DataResource> target;

    public NonOverwritableConsumer(Map<DataKey, DataResource> target) {
      this.target = target;
    }

    @Override
    public void consume(DataKey key, DataResource value) {
      if (!target.containsKey(key)) {
        target.put(key, value);
      }
    }
  }

  @VisibleForTesting
  static class OverwritableConsumer<K extends DataKey, V extends DataValue>
      implements KeyValueConsumer<K, V> {
    private Map<K, V> target;
    private Set<MergeConflict> conflicts;

    OverwritableConsumer(Map<K, V> target, Set<MergeConflict> conflicts) {
      this.target = target;
      this.conflicts = conflicts;
    }

    @Override
    public void consume(K key, V value) {
      if (target.containsKey(key)) {
        conflicts.add(MergeConflict.between(key, value, target.get(key)));
      } else {
        target.put(key, value);
      }
    }
  }

  /**
   * An AndroidDataPathWalker that collects DataAsset and DataResources for an ParsedAndroidData.
   */
  private static final class ParsedAndroidDataBuildingPathWalker implements AndroidDataPathWalker {
    private final Set<MergeConflict> conflicts;
    private final Map<DataKey, DataAsset> assets;
    private final ResourceFileVisitor resourceVisitor;
    private static final ImmutableSet<FileVisitOption> FOLLOW_LINKS =
        ImmutableSet.of(FileVisitOption.FOLLOW_LINKS);
    private Map<DataKey, DataResource> overwritingResources;
    private Map<DataKey, DataResource> nonOverwritingResources;

    private static ParsedAndroidDataBuildingPathWalker create() {
      final Map<DataKey, DataResource> overwritingResources = new HashMap<>();
      final Map<DataKey, DataResource> nonOverwritingResources = new HashMap<>();
      final Map<DataKey, DataAsset> assets = new HashMap<>();
      final Set<MergeConflict> conflicts = new HashSet<>();

      final ResourceFileVisitor resourceVisitor =
          new ResourceFileVisitor(
              new OverwritableConsumer<>(overwritingResources, conflicts),
              new NonOverwritableConsumer(nonOverwritingResources));
      return new ParsedAndroidDataBuildingPathWalker(
          conflicts, assets, overwritingResources, nonOverwritingResources, resourceVisitor);
    }

    private ParsedAndroidDataBuildingPathWalker(
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
          new AssetFileVisitor(
              RelativeAssetPath.Factory.of(path), new OverwritableConsumer<>(assets, conflicts)));
    }

    /**
     * Creates an {@link ParsedAndroidData} from {@link DataAsset} and {@link DataResource}.
     */
    public ParsedAndroidData createParsedAndroidData() throws MergingException {
      resourceVisitor.checkForErrors();
      return ParsedAndroidData.of(
          ImmutableSet.copyOf(conflicts),
          ImmutableMap.copyOf(overwritingResources),
          ImmutableMap.copyOf(nonOverwritingResources),
          ImmutableMap.copyOf(assets));
    }
  }

  /**
   * A {@link java.nio.file.FileVisitor} that walks the asset tree and extracts {@link DataAsset}s.
   */
  private static class AssetFileVisitor extends SimpleFileVisitor<Path> {
    private final RelativeAssetPath.Factory dataKeyFactory;
    private KeyValueConsumer<DataKey, DataAsset> assetConsumer;

    public AssetFileVisitor(
        RelativeAssetPath.Factory dataKeyFactory,
        KeyValueConsumer<DataKey, DataAsset> assetConsumer) {
      this.dataKeyFactory = dataKeyFactory;
      this.assetConsumer = assetConsumer;
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
      if (!Files.isDirectory(path)) {
        RelativeAssetPath key = dataKeyFactory.create(path);
        DataValueFile asset = DataValueFile.of(path);
        assetConsumer.consume(key, asset);
      }
      return super.visitFile(path, attrs);
    }
  }

  /**
   * A FileVisitor that walks a resource tree and extract FullyQualifiedName and resource values.
   */
  private static class ResourceFileVisitor extends SimpleFileVisitor<Path> {
    private final List<Exception> errors = new ArrayList<>();
    private final OverwritableConsumer<DataKey, DataResource> overwritingConsumer;
    private final NonOverwritableConsumer nonOverwritingConsumer;
    private boolean inValuesSubtree;
    private FullyQualifiedName.Factory fqnFactory;
    private final XMLInputFactory xmlInputFactory = XMLInputFactory.newFactory();

    public ResourceFileVisitor(
        OverwritableConsumer<DataKey, DataResource> overwritingConsumer,
        NonOverwritableConsumer nonOverwritingConsumer) {
      this.overwritingConsumer = overwritingConsumer;
      this.nonOverwritingConsumer = nonOverwritingConsumer;
    }

    private static String deriveRawFullyQualifiedName(Path path) {
      if (path.getNameCount() < 2) {
        throw new IllegalArgumentException(
            String.format(
                "The resource path %s is too short. "
                    + "The path is expected to be <resource type>/<file name>.",
                path));
      }
      String pathWithExtension =
          path.subpath(path.getNameCount() - 2, path.getNameCount()).toString();
      int extensionStart = pathWithExtension.lastIndexOf('.');
      if (extensionStart > 0) {
        return pathWithExtension.substring(0, extensionStart);
      }
      return pathWithExtension;
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
            DataResourceXml.parse(
                xmlInputFactory, path, fqnFactory, overwritingConsumer, nonOverwritingConsumer);
          } else {
            String rawFqn = deriveRawFullyQualifiedName(path);
            FullyQualifiedName key = fqnFactory.parse(rawFqn);
            overwritingConsumer.consume(key, DataValueFile.of(path));
          }
        }
      } catch (IllegalArgumentException | XMLStreamException e) {
        errors.add(e);
      }
      return super.visitFile(path, attrs);
    }

    public List<Exception> getErrors() {
      return errors;
    }
  }

  /** Creates ParsedAndroidData of conflicts, assets overwriting and nonOverwriting resources. */
  public static ParsedAndroidData of(
      ImmutableSet<MergeConflict> conflicts,
      ImmutableMap<DataKey, DataResource> overwritingResources,
      ImmutableMap<DataKey, DataResource> nonOverwritingResources,
      ImmutableMap<DataKey, DataAsset> assets) {
    return new ParsedAndroidData(conflicts, overwritingResources, nonOverwritingResources, assets);
  }

  /**
   * Creates an ParsedAndroidData from an UnvalidatedAndroidData.
   *
   * The adding process parses out all the provided symbol into DataResources and DataAssets
   * objects.
   *
   * @param primary The primary data to parse into DataResources and DataAssets.
   * @throws IOException when there are issues with reading files.
   * @throws MergingException when there is invalid resource information.
   */
  public static ParsedAndroidData from(UnvalidatedAndroidData primary)
      throws IOException, MergingException {
    final ParsedAndroidDataBuildingPathWalker pathWalker =
        ParsedAndroidDataBuildingPathWalker.create();
    primary.walk(pathWalker);
    return pathWalker.createParsedAndroidData();
  }

  /**
   * Creates an ParsedAndroidData from a list of DependencyAndroidData instances.
   *
   * The adding process parses out all the provided symbol into DataResources and DataAssets
   * objects.
   *
   * @param dependencyAndroidDataList The dependency data to parse into DataResources and
   *        DataAssets.
   * @throws IOException when there are issues with reading files.
   * @throws MergingException when there is invalid resource information.
   */
  public static ParsedAndroidData from(List<DependencyAndroidData> dependencyAndroidDataList)
      throws IOException, MergingException {
    final ParsedAndroidDataBuildingPathWalker pathWalker =
        ParsedAndroidDataBuildingPathWalker.create();
    for (DependencyAndroidData data : dependencyAndroidDataList) {
      data.walk(pathWalker);
    }
    return pathWalker.createParsedAndroidData();
  }

  private final ImmutableSet<MergeConflict> conflicts;
  private final ImmutableMap<DataKey, DataResource> overwritingResources;
  private final ImmutableMap<DataKey, DataResource> nonOverwritingResources;
  private final ImmutableMap<DataKey, DataAsset> assets;

  private ParsedAndroidData(
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
    if (!(other instanceof ParsedAndroidData)) {
      return false;
    }
    ParsedAndroidData that = (ParsedAndroidData) other;
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

  Iterable<Entry<DataKey, DataResource>> iterateOverwritableEntries() {
    return overwritingResources.entrySet();
  }

  public Iterable<Entry<DataKey, DataResource>> iterateDataResourceEntries() {
    return Iterables.concat(overwritingResources.entrySet(), nonOverwritingResources.entrySet());
  }

  boolean containsAsset(DataKey name) {
    return assets.containsKey(name);
  }

  Iterable<Entry<DataKey, DataAsset>> iterateAssetEntries() {
    return assets.entrySet();
  }

  MergeConflict foundResourceConflict(DataKey key, DataResource value) {
    return MergeConflict.between(key, overwritingResources.get(key), value);
  }

  MergeConflict foundAssetConflict(DataKey key, DataAsset value) {
    return MergeConflict.between(key, assets.get(key), value);
  }

  ImmutableMap<DataKey, DataResource> mergeNonOverwritable(ParsedAndroidData other) {
    Map<DataKey, DataResource> merged = new HashMap<>(other.nonOverwritingResources);
    merged.putAll(nonOverwritingResources);
    return ImmutableMap.copyOf(merged);
  }

  ImmutableSet<MergeConflict> conflicts() {
    return conflicts;
  }
}
