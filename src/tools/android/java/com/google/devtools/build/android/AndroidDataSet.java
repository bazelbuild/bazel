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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
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
import java.util.List;
import java.util.Objects;

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
    private final List<DataAsset> assets;
    private final ResourceFileVisitor resourceVisitor;
    private static final ImmutableSet<FileVisitOption> FOLLOW_LINKS =
        ImmutableSet.of(FileVisitOption.FOLLOW_LINKS);
    private List<DataResource> overwritingResources;
    private List<DataResource> nonOverwritingResources;

    private static AndroidDataSetBuildingPathWalker create() {
      List<DataResource> overwritingResources = new ArrayList<>();
      List<DataResource> nonOverwritingResources = new ArrayList<>();
      final List<DataAsset> assets = new ArrayList<>();
      final ResourceFileVisitor resourceVisitor =
          new ResourceFileVisitor(overwritingResources, nonOverwritingResources);
      return new AndroidDataSetBuildingPathWalker(
          assets, overwritingResources, nonOverwritingResources, resourceVisitor);
    }

    private AndroidDataSetBuildingPathWalker(
        List<DataAsset> assets,
        List<DataResource> overwritingResources,
        List<DataResource> nonOverwritingResources,
        ResourceFileVisitor resourceVisitor) {
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
          new AssetFileVisitor(RelativeAssetPath.Factory.of(path), assets));
    }

    /**
     * Creates an AndroidDataSet from the collected DataAsset and DataResource instances.
     */
    public AndroidDataSet createAndroidDataSet() throws MergingException {
      resourceVisitor.checkForErrors();
      return AndroidDataSet.of(overwritingResources, nonOverwritingResources, assets);
    }
  }

  /**
   * A FileVisitor that walks the asset tree and extracts FileDataResources.
   */
  private static class AssetFileVisitor extends SimpleFileVisitor<Path> {
    private final RelativeAssetPath.Factory dataKeyFactory;
    private final List<DataAsset> assets;

    public AssetFileVisitor(RelativeAssetPath.Factory dataKeyFactory, List<DataAsset> assets) {
      this.dataKeyFactory = dataKeyFactory;
      this.assets = assets;
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
      if (!Files.isDirectory(path)) {
        assets.add(FileDataResource.of(dataKeyFactory.create(path), path));
      }
      return super.visitFile(path, attrs);
    }
  }

  /**
   * A FileVisitor that walks a resource tree and extract FullyQualifiedName and resource values.
   */
  private static class ResourceFileVisitor extends SimpleFileVisitor<Path> {
    private final List<DataResource> overwritingResources;
    private final List<DataResource> nonOverwritingResources;
    private final List<Exception> errors = new ArrayList<>();
    private boolean inValuesSubtree;
    private FullyQualifiedName.Factory fqnFactory;
    private final XMLInputFactory xmlInputFactory = XMLInputFactory.newFactory();

    public ResourceFileVisitor(
        List<DataResource> overwritingResources, List<DataResource> nonOverwritingResources) {
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
            XmlDataResource.fromPath(
                xmlInputFactory, path, fqnFactory, overwritingResources, nonOverwritingResources);
          } else {
            overwritingResources.add(FileDataResource.fromPath(path, fqnFactory));
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

  /** Creates an AndroidDataSet of the overwriting, nonOverwritingResources, and asset lists. */
  public static AndroidDataSet of(
      List<DataResource> overwritingResources,
      List<DataResource> nonOverwritingResources,
      List<DataAsset> assets) {
    return new AndroidDataSet(
        ImmutableList.copyOf(overwritingResources),
        ImmutableList.copyOf(nonOverwritingResources),
        ImmutableList.copyOf(assets));
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

  private final ImmutableList<DataResource> overwritingResources;
  private final ImmutableList<DataResource> nonOverwritingResources;
  private final ImmutableList<DataAsset> assets;

  private AndroidDataSet(
      ImmutableList<DataResource> overwritingResources,
      ImmutableList<DataResource> nonOverwritingResources,
      ImmutableList<DataAsset> assets) {
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
        && Objects.equals(assets, that.assets);
  }

  @Override
  public int hashCode() {
    return Objects.hash(overwritingResources, nonOverwritingResources, assets);
  }

  /**
   * Returns a list of resources that would overwrite other values when defined.
   *
   * <p>Example:
   *
   * A string resource (string.Foo=bar) could be redefined at string.Foo=baz.
   *
   * @return A list of overwriting resources.
   */
  public List<DataResource> getOverwritingResources() {
    return overwritingResources;
  }

  /**
   * Returns a list of resources that would not overwrite other values when defined.
   *
   * <p>Example:
   *
   * A id resource (id.Foo) could be redefined at id.Foo with no adverse effects.
   *
   * @return A list of non-overwriting resources.
   */
  public List<DataResource> getNonOverwritingResources() {
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
   * @return A list of overwriting assets.
   */
  public List<DataAsset> getAssets() {
    return assets;
  }
}
