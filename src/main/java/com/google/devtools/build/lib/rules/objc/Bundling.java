// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.NESTED_BUNDLE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;

/**
 * Contains information regarding the creation of an iOS bundle.
 */
@Immutable
final class Bundling {
  static final class Builder {
    private String name;
    private String bundleDirFormat;
    private ImmutableList<BundleableFile> extraBundleFiles = ImmutableList.of();
    private ObjcProvider objcProvider;
    private InfoplistMerging infoplistMerging;
    private IntermediateArtifacts intermediateArtifacts;
    private String primaryBundleId;
    private String fallbackBundleId;
    private String architecture;

    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    /**
     * Sets the CPU architecture this bundling was constructed for. Legal value are any that may be
     * set on {@link ObjcConfiguration#getIosCpu()}.
     */
    public Builder setArchitecture(String architecture) {
      this.architecture = architecture;
      return this;
    }

    public Builder setBundleDirFormat(String bundleDirFormat) {
      this.bundleDirFormat = bundleDirFormat;
      return this;
    }

    public Builder setExtraBundleFiles(ImmutableList<BundleableFile> extraBundleFiles) {
      this.extraBundleFiles = extraBundleFiles;
      return this;
    }

    public Builder setObjcProvider(ObjcProvider objcProvider) {
      this.objcProvider = objcProvider;
      return this;
    }

    public Builder setInfoplistMerging(InfoplistMerging infoplistMerging) {
      this.infoplistMerging = infoplistMerging;
      return this;
    }

    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }
    
    public Builder setPrimaryBundleId(String primaryId) {
      this.primaryBundleId = primaryId;
      return this;
    }
    
    public Builder setFallbackBundleId(String fallbackId) {
      this.fallbackBundleId = fallbackId;
      return this;
    }

    private static NestedSet<Artifact> nestedBundleContentArtifacts(Iterable<Bundling> bundles) {
      NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.stableOrder();
      for (Bundling bundle : bundles) {
        artifacts.addTransitive(bundle.getBundleContentArtifacts());
      }
      return artifacts.build();
    }

    public Bundling build() {
      Preconditions.checkNotNull(intermediateArtifacts, "intermediateArtifacts");

      Optional<Artifact> actoolzipOutput = Optional.absent();
      if (!Iterables.isEmpty(objcProvider.get(ASSET_CATALOG))) {
        actoolzipOutput = Optional.of(intermediateArtifacts.actoolzipOutput());
      }

      Optional<Artifact> combinedArchitectureBinary = Optional.absent();
      if (!Iterables.isEmpty(objcProvider.get(LIBRARY))
          || !Iterables.isEmpty(objcProvider.get(IMPORTED_LIBRARY))) {
        combinedArchitectureBinary =
            Optional.of(intermediateArtifacts.combinedArchitectureBinary());
      }

      NestedSet<Artifact> mergeZips = NestedSetBuilder.<Artifact>stableOrder()
          .addAll(actoolzipOutput.asSet())
          .addTransitive(objcProvider.get(MERGE_ZIP))
          .build();
      NestedSet<Artifact> bundleContentArtifacts = NestedSetBuilder.<Artifact>stableOrder()
          .addTransitive(nestedBundleContentArtifacts(objcProvider.get(NESTED_BUNDLE)))
          .addAll(combinedArchitectureBinary.asSet())
          .addAll(infoplistMerging.getPlistWithEverything().asSet())
          .addTransitive(mergeZips)
          .addAll(BundleableFile.toArtifacts(extraBundleFiles))
          .addAll(BundleableFile.toArtifacts(objcProvider.get(BUNDLE_FILE)))
          .addAll(Xcdatamodel.outputZips(objcProvider.get(XCDATAMODEL)))
          .build();

      return new Bundling(name, bundleDirFormat, combinedArchitectureBinary, extraBundleFiles,
          objcProvider, infoplistMerging, actoolzipOutput, bundleContentArtifacts, mergeZips, 
          primaryBundleId, fallbackBundleId, architecture);
    }
  }

  private final String name;
  private final String architecture;
  private final String bundleDirFormat;
  private final Optional<Artifact> combinedArchitectureBinary;
  private final ImmutableList<BundleableFile> extraBundleFiles;
  private final ObjcProvider objcProvider;
  private final InfoplistMerging infoplistMerging;
  private final Optional<Artifact> actoolzipOutput;
  private final NestedSet<Artifact> bundleContentArtifacts;
  private final NestedSet<Artifact> mergeZips;
  private final String primaryBundleId;
  private final String fallbackBundleId;

  private Bundling(
      String name,
      String bundleDirFormat,
      Optional<Artifact> combinedArchitectureBinary,
      ImmutableList<BundleableFile> extraBundleFiles,
      ObjcProvider objcProvider,
      InfoplistMerging infoplistMerging,
      Optional<Artifact> actoolzipOutput,
      NestedSet<Artifact> bundleContentArtifacts,
      NestedSet<Artifact> mergeZips,
      String primaryBundleId,
      String fallbackBundleId,
      String architecture) {
    this.name = Preconditions.checkNotNull(name);
    this.bundleDirFormat = Preconditions.checkNotNull(bundleDirFormat);
    this.combinedArchitectureBinary = Preconditions.checkNotNull(combinedArchitectureBinary);
    this.extraBundleFiles = Preconditions.checkNotNull(extraBundleFiles);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.infoplistMerging = Preconditions.checkNotNull(infoplistMerging);
    this.actoolzipOutput = Preconditions.checkNotNull(actoolzipOutput);
    this.bundleContentArtifacts = Preconditions.checkNotNull(bundleContentArtifacts);
    this.mergeZips = Preconditions.checkNotNull(mergeZips);
    this.fallbackBundleId = fallbackBundleId;
    this.primaryBundleId = primaryBundleId;
    this.architecture = Preconditions.checkNotNull(architecture);
  }

  /**
   * The bundle directory. For apps, this would be {@code "Payload/TARGET_NAME.app"}, which is where
   * in the bundle zip archive every file is found, including the linked binary, nested bundles, and
   * everything returned by {@link #getExtraBundleFiles()}.
   */
  public String getBundleDir() {
    return String.format(bundleDirFormat, name);
  }

  /**
   * The name of the bundle, from which the bundle root and the path of the linked binary in the
   * bundle archive are derived.
   */
  public String getName() {
    return name;
  }

  /**
   * An {@link Optional} with the linked binary artifact, or {@link Optional#absent()} if it is
   * empty and should not be included in the bundle.
   */
  public Optional<Artifact> getCombinedArchitectureBinary() {
    return combinedArchitectureBinary;
  }

  /**
   * Extra bundle files to include in the bundle which are not automatically deduced by the contents
   * of the provider. These files are placed under the bundle root (possibly nested, of course,
   * depending on the bundle path of the files).
   */
  public ImmutableList<BundleableFile> getExtraBundleFiles() {
    return extraBundleFiles;
  }

  /**
   * The {@link ObjcProvider} for this bundle.
   */
  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  /**
   * Information on the Info.plist and its merge inputs for this bundle. Note that an infoplist is
   * only included in the bundle if it has one or more merge inputs.
   */
  public InfoplistMerging getInfoplistMerging() {
    return infoplistMerging;
  }

  /**
   * The location of the actoolzip output for this bundle. This is non-absent only included in the
   * bundle if there is at least one asset catalog artifact supplied by
   * {@link ObjcProvider#ASSET_CATALOG}.
   */
  public Optional<Artifact> getActoolzipOutput() {
    return actoolzipOutput;
  }

  /**
   * Returns all zip files whose contents should be merged into this bundle under the main bundle
   * directory. For instance, if a merge zip contains files a/b and c/d, then the resulting bundling
   * would have additional files at:
   * <ul>
   *   <li>{bundleDir}/a/b
   *   <li>{bundleDir}/c/d
   * </ul>
   */
  public NestedSet<Artifact> getMergeZips() {
    return mergeZips;
  }

  /**
   * Returns the variable substitutions that should be used when merging the plist info file of
   * this bundle.
   */
  public Map<String, String> variableSubstitutions() {
    return ImmutableMap.of(
        "EXECUTABLE_NAME", name,
        "BUNDLE_NAME", new PathFragment(getBundleDir()).getBaseName(),
        "PRODUCT_NAME", name);
  }

  /**
   * Returns the artifacts that are required to generate this bundle.
   */
  public NestedSet<Artifact> getBundleContentArtifacts() {
    return bundleContentArtifacts;
  }
  
  /**
   * Returns primary bundle ID to use, can be null.
   */
  public String getPrimaryBundleId() {
    return primaryBundleId;
  }
  
  /**
   * Returns fallback bundle ID to use when primary isn't set.
   */
  public String getFallbackBundleId() {
    return fallbackBundleId;
  }

  /**
   * Returns the iOS CPU architecture this bundle was constructed for.
   */
  public String getArchitecture() {
    return architecture;
  }
}
