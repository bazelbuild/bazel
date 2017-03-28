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

import com.android.SdkConstants;
import com.android.ide.common.res2.MergingException;
import com.android.resources.FolderTypeRelationship;
import com.android.resources.ResourceFolderType;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;
import javax.annotation.concurrent.Immutable;
import javax.annotation.concurrent.NotThreadSafe;
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
  private static final Logger logger = Logger.getLogger(ParsedAndroidData.class.getCanonicalName());

  @NotThreadSafe
  static class Builder {
    private final Map<DataKey, DataResource> overwritingResources;
    private final Map<DataKey, DataResource> combiningResources;
    private final Map<DataKey, DataAsset> assets;
    private final Set<MergeConflict> conflicts;
    private final List<Exception> errors = new ArrayList<>();

    public Builder(
        Map<DataKey, DataResource> overwritingResources,
        Map<DataKey, DataResource> combiningResources,
        Map<DataKey, DataAsset> assets,
        Set<MergeConflict> conflicts) {
      this.overwritingResources = overwritingResources;
      this.combiningResources = combiningResources;
      this.assets = assets;
      this.conflicts = conflicts;
    }

    static Builder newBuilder() {
      final Map<DataKey, DataResource> overwritingResources = new LinkedHashMap<>();
      final Map<DataKey, DataResource> combiningResources = new LinkedHashMap<>();
      final Map<DataKey, DataAsset> assets = new LinkedHashMap<>();
      final Set<MergeConflict> conflicts = new LinkedHashSet<>();
      return new Builder(overwritingResources, combiningResources, assets, conflicts);
    }

    private void checkForErrors() throws MergingException {
      if (!errors.isEmpty()) {
        MergingException mergingException =
             MergingException
                 .withMessage(String.format("%s Parse Error(s)", errors.size())).build();
        for (Exception e : errors) {
          mergingException.addSuppressed(e);
        }
        throw mergingException;
      }
    }

    ParsedAndroidData build() throws MergingException {
      checkForErrors();
      return ParsedAndroidData.of(
          ImmutableSet.copyOf(conflicts),
          ImmutableMap.copyOf(overwritingResources),
          ImmutableMap.copyOf(combiningResources),
          ImmutableMap.copyOf(assets));
    }

    /** Copies the data to the targetBuilder from the current builder. */
   public void copyTo(Builder targetBuilder) {
      KeyValueConsumers consumers = targetBuilder.consumers();
      for (Entry<DataKey, DataResource> entry : overwritingResources.entrySet()) {
        consumers.overwritingConsumer.consume(entry.getKey(), entry.getValue());
      }
      for (Entry<DataKey, DataResource> entry : combiningResources.entrySet()) {
        consumers.combiningConsumer.consume(entry.getKey(), entry.getValue());
      }
      for (Entry<DataKey, DataAsset> entry : assets.entrySet()) {
        consumers.assetConsumer.consume(entry.getKey(), entry.getValue());
      }
      targetBuilder.conflicts.addAll(conflicts);
    }

    ResourceFileVisitor resourceVisitor() {
      return new ResourceFileVisitor(
          new OverwritableConsumer<>(overwritingResources, conflicts),
          new CombiningConsumer(combiningResources),
          errors);
    }

    AssetFileVisitor assetVisitorFor(Path path) {
      return new AssetFileVisitor(
          RelativeAssetPath.Factory.of(path), new OverwritableConsumer<>(assets, conflicts));
    }

    public KeyValueConsumers consumers() {
      return KeyValueConsumers.of(
          new OverwritableConsumer<>(overwritingResources, conflicts),
          new CombiningConsumer(combiningResources),
          new OverwritableConsumer<>(assets, conflicts));
    }
  }

  /** A Consumer style interface that will appendTo a DataKey and DataValue. */
  interface KeyValueConsumer<K extends DataKey, V extends DataValue> {
    void consume(K key, V value);
  }

  @VisibleForTesting
  static class CombiningConsumer implements KeyValueConsumer<DataKey, DataResource> {

    private Map<DataKey, DataResource> target;

    CombiningConsumer(Map<DataKey, DataResource> target) {
      this.target = target;
    }

    @Override
    public void consume(DataKey key, DataResource value) {
      if (target.containsKey(key)) {
        target.put(key, target.get(key).combineWith(value));
      } else {
        target.put(key, value);
      }
    }
  }

  @VisibleForTesting
  static class OverwritableConsumer<K extends DataKey, V extends DataValue>
      implements KeyValueConsumer<K, V> {
    private final Map<K, V> target;
    private final Set<MergeConflict> conflicts;
    private final boolean recordConflicts;

    OverwritableConsumer(Map<K, V> target, Set<MergeConflict> conflicts, boolean recordConflicts) {
      this.target = target;
      this.conflicts = conflicts;
      this.recordConflicts = recordConflicts;
    }

    OverwritableConsumer(Map<K, V> target, Set<MergeConflict> conflicts) {
      this(target, conflicts, true);
    }

    @Override
    public void consume(K key, V value) {
      if (target.containsKey(key)) {
        V other = target.get(key);
        if (other.source().hasOveridden(value.source())) {
          // technically a noop, but this complicated enough to explicit.
          target.put(key, other);
        } else if (value.source().hasOveridden(other.source())) {
          target.put(key, value);
        } else {
          target.put(key, overwrite(key, value, other));
        }
      } else {
        target.put(key, value);
      }
    }

    private V overwrite(K key, V overwriter, V overwritee) {
      // TODO(corysmith): Cleanup type system.
      @SuppressWarnings("unchecked")
      V updated = (V) overwriter.update(overwriter.source().overwrite(overwritee.source()));
      if (recordConflicts) {
        conflicts.add(MergeConflict.between(key, updated, overwritee));
      }
      return updated;
    }
  }

  /** An AndroidDataPathWalker that collects DataAsset and DataResources for a ParsedAndroidData. */
  static final class ParsedAndroidDataBuildingPathWalker implements AndroidDataPathWalker {
    private static final ImmutableSet<FileVisitOption> FOLLOW_LINKS =
        ImmutableSet.of(FileVisitOption.FOLLOW_LINKS);
    private final Builder builder;

    private ParsedAndroidDataBuildingPathWalker(Builder builder) {
      this.builder = builder;
    }

    static ParsedAndroidDataBuildingPathWalker create(Builder builder) {
      return new ParsedAndroidDataBuildingPathWalker(builder);
    }

    @Override
    public void walkResources(Path path) throws IOException {
      Files.walkFileTree(path, FOLLOW_LINKS, Integer.MAX_VALUE, builder.resourceVisitor());
    }

    @Override
    public void walkAssets(Path path) throws IOException {
      Files.walkFileTree(path, FOLLOW_LINKS, Integer.MAX_VALUE, builder.assetVisitorFor(path));
    }

    ParsedAndroidData createParsedAndroidData() throws MergingException {
      return builder.build();
    }
  }

  /**
   * A {@link java.nio.file.FileVisitor} that walks the asset tree and extracts {@link DataAsset}s.
   */
  private static class AssetFileVisitor extends SimpleFileVisitor<Path> {
    private final RelativeAssetPath.Factory dataKeyFactory;
    private KeyValueConsumer<DataKey, DataAsset> assetConsumer;

    AssetFileVisitor(
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
    private final KeyValueConsumer<DataKey, DataResource> overwritingConsumer;
    private final KeyValueConsumer<DataKey, DataResource> combiningResources;
    private final List<Exception> errors;
    private ResourceFolderType folderType;
    private FullyQualifiedName.Factory fqnFactory;

    /**
     * Resource folders with XML files that may contain "@+id".
     * See android_ide_common's {@link FolderTypeRelationship}.
     */
    private static final EnumSet<ResourceFolderType> ID_PROVIDING_RESOURCE_TYPES = EnumSet.of(
        ResourceFolderType.DRAWABLE,
        ResourceFolderType.LAYOUT,
        ResourceFolderType.MENU,
        ResourceFolderType.TRANSITION,
        ResourceFolderType.XML);

    ResourceFileVisitor(
        KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
        KeyValueConsumer<DataKey, DataResource> combiningResources,
        List<Exception> errors) {
      this.overwritingConsumer = overwritingConsumer;
      this.combiningResources = combiningResources;
      this.errors = errors;
    }

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs)
        throws IOException {
      final String[] dirNameAndQualifiers = dir.getFileName().toString().split(
          SdkConstants.RES_QUALIFIER_SEP);
      folderType = ResourceFolderType.getTypeByName(dirNameAndQualifiers[0]);
      if (folderType == null) {
        return FileVisitResult.CONTINUE;
      }
      try {
        fqnFactory = FullyQualifiedName.Factory.fromDirectoryName(dirNameAndQualifiers);
        return FileVisitResult.CONTINUE;
      } catch (IllegalArgumentException e) {
        logger.warning(
            String.format("%s is an invalid resource directory due to %s", dir, e.getMessage()));
        return FileVisitResult.SKIP_SUBTREE;
      }
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
      try {
        if (!Files.isDirectory(path) && !path.getFileName().toString().startsWith(".")) {
          if (folderType == ResourceFolderType.VALUES) {
            DataResourceXml.parse(
                XmlResourceValues.getXmlInputFactory(),
                path,
                fqnFactory,
                overwritingConsumer,
                combiningResources);
          } else if (folderType != null) {
            FullyQualifiedName key = fqnFactory.parse(path);
            if (ID_PROVIDING_RESOURCE_TYPES.contains(folderType)
                && path.getFileName().toString().endsWith(SdkConstants.DOT_XML)) {
              DataValueFileWithIds.parse(
                  XmlResourceValues.getXmlInputFactory(),
                  path,
                  key,
                  fqnFactory,
                  overwritingConsumer,
                  combiningResources);
            } else {
              overwritingConsumer.consume(key, DataValueFile.of(path));
            }
          }
        }
      } catch (IllegalArgumentException | XMLStreamException e) {
        errors.add(e);
      }
      return super.visitFile(path, attrs);
    }
  }

  /** Creates ParsedAndroidData of conflicts, assets overwriting and combining resources. */
  public static ParsedAndroidData of(
      ImmutableSet<MergeConflict> conflicts,
      ImmutableMap<DataKey, DataResource> overwritingResources,
      ImmutableMap<DataKey, DataResource> combiningResources,
      ImmutableMap<DataKey, DataAsset> assets) {
    return new ParsedAndroidData(conflicts, overwritingResources, combiningResources, assets);
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
  public static ParsedAndroidData from(UnvalidatedAndroidDirectories primary)
      throws IOException, MergingException {
    final ParsedAndroidDataBuildingPathWalker pathWalker =
        ParsedAndroidDataBuildingPathWalker.create(Builder.newBuilder());
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
        ParsedAndroidDataBuildingPathWalker.create(Builder.newBuilder());
    for (DependencyAndroidData data : dependencyAndroidDataList) {
      data.walk(pathWalker);
    }
    return pathWalker.createParsedAndroidData();
  }

  private final ImmutableSet<MergeConflict> conflicts;
  private final ImmutableMap<DataKey, DataResource> overwritingResources;
  private final ImmutableMap<DataKey, DataResource> combiningResources;
  private final ImmutableMap<DataKey, DataAsset> assets;

  private ParsedAndroidData(
      ImmutableSet<MergeConflict> conflicts,
      ImmutableMap<DataKey, DataResource> overwritingResources,
      ImmutableMap<DataKey, DataResource> combiningResources,
      ImmutableMap<DataKey, DataAsset> assets) {
    this.conflicts = conflicts;
    this.overwritingResources = overwritingResources;
    this.combiningResources = combiningResources;
    this.assets = assets;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("overwritingResources", overwritingResources)
        .add("combiningResources", combiningResources)
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
        && Objects.equals(combiningResources, that.combiningResources)
        && Objects.equals(conflicts, that.conflicts)
        && Objects.equals(assets, that.assets);
  }

  @Override
  public int hashCode() {
    return Objects.hash(conflicts, overwritingResources, combiningResources, assets);
  }

  /**
   * Returns a list of resources that would overwrite other values when defined.
   *
   * <p>
   * Example:
   *
   * A string resource (string.Foo=bar) could be redefined at string.Foo=baz.
   *
   * @return A map of key -&gt; overwriting resources.
   */
  @VisibleForTesting
  Map<DataKey, DataResource> getOverwritingResources() {
    return overwritingResources;
  }

  /**
   * Returns a list of resources are combined with other values that have the same key.
   *
   * <p>
   * Example:
   *
   * A id resource (id.Foo) combined id.Foo with no adverse effects, whereas two stylable.Bar
   * resources would be combined, resulting in a Styleable containing a union of the attributes.
   * See {@link StyleableXmlResourceValue} for more information.
   *
   * @return A map of key -&gt; combing resources.
   */
  @VisibleForTesting
  Map<DataKey, DataResource> getCombiningResources() {
    return combiningResources;
  }

  /**
   * Returns a list of assets.
   *
   * Assets always overwrite during merging, just like overwriting resources.
   * <p>
   * Example:
   *
   * A text asset (foo/bar.txt, containing fooza) could be replaced with (foo/bar.txt, containing
   * ouza!) depending on the merging process.
   *
   * @return A map of key -&gt; assets.
   */
  public Map<DataKey, DataAsset> getAssets() {
    return assets;
  }

  public boolean containsOverwritable(DataKey name) {
    return overwritingResources.containsKey(name);
  }

  public boolean containsCombineable(DataKey key) {
    return combiningResources.containsKey(key);
  }

  public DataResource getOverwritable(DataKey name) {
    return overwritingResources.get(name);
  }

  void writeResourcesTo(AndroidResourceClassWriter writer) {
    for (Entry<DataKey, DataResource> resource : iterateDataResourceEntries()) {
      resource.getValue().writeResourceToClass((FullyQualifiedName) resource.getKey(), writer);
    }
  }

  void writeResourcesTo(AndroidDataWriter writer) throws MergingException {
    for (Entry<DataKey, DataResource> resource : iterateDataResourceEntries()) {
      resource.getValue().writeResource((FullyQualifiedName) resource.getKey(), writer);
    }
  }
  
  void serializeResourcesTo(AndroidDataSerializer serializer) {
    for (Entry<DataKey, DataResource> resource : iterateDataResourceEntries()) {
      serializer.queueForSerialization(resource.getKey(), resource.getValue());
    }
  }

  void writeAssetsTo(AndroidDataWriter writer) throws IOException {
    for (Entry<DataKey, DataAsset> resource : iterateAssetEntries()) {
      resource.getValue().writeAsset((RelativeAssetPath) resource.getKey(), writer);
    }
  }
 
  void serializeAssetsTo(AndroidDataSerializer serializer) {
    for (Entry<DataKey, DataAsset> resource : iterateAssetEntries()) {
      serializer.queueForSerialization(resource.getKey(), resource.getValue());
    }
  }

  Iterable<Entry<DataKey, DataResource>> iterateOverwritableEntries() {
    return overwritingResources.entrySet();
  }

  ParsedAndroidData overwrite(ParsedAndroidData overwritableData, boolean createConflicts) {
    Map<DataKey, DataResource> newEntries = new LinkedHashMap<>();
    Set<MergeConflict> newConflicts =
        createConflicts ? new LinkedHashSet<MergeConflict>() : conflicts;
    overwrite(
        overwritableData.overwritingResources,
        overwritingResources,
        new OverwritableConsumer<>(newEntries, newConflicts));

    Map<DataKey, DataAsset> newAssets = new LinkedHashMap<>();
    overwrite(overwritableData.assets, assets, new OverwritableConsumer<>(newAssets, newConflicts));

    return ParsedAndroidData.of(
        ImmutableSet.copyOf(newConflicts),
        ImmutableMap.copyOf(newEntries),
        combiningResources,
        ImmutableMap.copyOf(newAssets));
  }

  private static <K extends DataKey, V extends DataValue> void overwrite(
      Map<K, V> overwritee, Map<K, V> overwriter, OverwritableConsumer<K, V> consumer) {
    SetView<K> overwritten = Sets.intersection(overwritee.keySet(), overwriter.keySet());

    // Feed the consumer keys and values that will be overwritten, followed by the overwritting
    // value. This ensures the proper book keeping is done inside the consumer.
    for (K key : overwritten) {
      consumer.consume(key, overwritee.get(key));
    }
    for (K key : overwriter.keySet()) {
      consumer.consume(key, overwriter.get(key));
    }
  }

  /** Combines all combinable resources. */
  ParsedAndroidData combine(ParsedAndroidData other) {
    Map<DataKey, DataResource> combinedResources = new LinkedHashMap<>();
    CombiningConsumer consumer = new CombiningConsumer(combinedResources);
    for (Entry<DataKey, DataResource> entry :
        Iterables.concat(combiningResources.entrySet(), other.combiningResources.entrySet())) {
      consumer.consume(entry.getKey(), entry.getValue());
    }
    return of(conflicts, overwritingResources, ImmutableMap.copyOf(combinedResources), assets);
  }

  /** Removes conflicts, resources, and assets that are in the other. */
  ParsedAndroidData difference(ParsedAndroidData other) {
    return of(
        ImmutableSet.copyOf(Sets.difference(conflicts, other.conflicts)),
        ImmutableMap.copyOf(
            Maps.difference(overwritingResources, other.overwritingResources).entriesOnlyOnLeft()),
        ImmutableMap.copyOf(
            Maps.difference(combiningResources, other.combiningResources).entriesOnlyOnLeft()),
        ImmutableMap.copyOf(Maps.difference(assets, other.assets).entriesOnlyOnLeft()));
  }

  /** Creates a union of both sets. Duplicates are ignored. */
  ParsedAndroidData union(ParsedAndroidData other) {
    return of(
        ImmutableSet.copyOf(Sets.union(conflicts, other.conflicts)),
        ImmutableMap.copyOf(
            Iterables.concat(
                overwritingResources.entrySet(), other.overwritingResources.entrySet())),
        ImmutableMap.copyOf(
            Iterables.concat(combiningResources.entrySet(), other.combiningResources.entrySet())),
        ImmutableMap.copyOf(Iterables.concat(assets.entrySet(), other.assets.entrySet())));
  }

  private Iterable<Entry<DataKey, DataResource>> iterateDataResourceEntries() {
    return Iterables.concat(overwritingResources.entrySet(), combiningResources.entrySet());
  }

  public Iterable<Entry<DataKey, DataResource>> iterateCombiningEntries() {
    return combiningResources.entrySet();
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

  ImmutableSet<MergeConflict> conflicts() {
    return conflicts;
  }

  public DataAsset getAsset(DataKey key) {
    return assets.get(key);
  }
}
