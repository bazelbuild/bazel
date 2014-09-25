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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Contains information regarding the creation of an iOS bundle.
 */
final class Bundling {
  static final class Builder {
    private String name;
    private Artifact linkedBinary;
    private ImmutableList<BundleableFile> extraBundleFiles;
    private ObjcProvider objcProvider;
    private InfoplistMerging infoplistMerging;
    private IntermediateArtifacts intermediateArtifacts;

    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    public Builder setLinkedBinary(Artifact linkedBinary) {
      this.linkedBinary = linkedBinary;
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

    public Bundling build() {
      Preconditions.checkNotNull(intermediateArtifacts, "intermediateArtifacts");

      Optional<Artifact> actoolzipOutput = Optional.absent();
      if (!Iterables.isEmpty(objcProvider.get(ASSET_CATALOG))) {
        actoolzipOutput = Optional.of(intermediateArtifacts.actoolzipOutput());
      }

      NestedSet<Artifact> bundleContentArtifacts = NestedSetBuilder.<Artifact>stableOrder()
          .add(linkedBinary)
          .addAll(infoplistMerging.getPlistWithEverything().asSet())
          .addAll(actoolzipOutput.asSet())
          .addAll(BundleableFile.toArtifacts(extraBundleFiles))
          .addAll(BundleableFile.toArtifacts(objcProvider.get(BUNDLE_FILE)))
          .addAll(Xcdatamodel.outputZips(objcProvider.get(XCDATAMODEL)))
          .build();

      return new Bundling(name, linkedBinary, extraBundleFiles, objcProvider,
          infoplistMerging, actoolzipOutput, bundleContentArtifacts);
    }
  }

  private final String name;
  private final Artifact linkedBinary;
  private final ImmutableList<BundleableFile> extraBundleFiles;
  private final ObjcProvider objcProvider;
  private final InfoplistMerging infoplistMerging;
  private final Optional<Artifact> actoolzipOutput;
  private final NestedSet<Artifact> bundleContentArtifacts;

  private Bundling(String name, Artifact linkedBinary,
      ImmutableList<BundleableFile> extraBundleFiles, ObjcProvider objcProvider,
      InfoplistMerging infoplistMerging, Optional<Artifact> actoolzipOutput,
      NestedSet<Artifact> bundleContentArtifacts) {
    this.name = Preconditions.checkNotNull(name);
    this.linkedBinary = Preconditions.checkNotNull(linkedBinary);
    this.extraBundleFiles = Preconditions.checkNotNull(extraBundleFiles);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.infoplistMerging = Preconditions.checkNotNull(infoplistMerging);
    this.actoolzipOutput = Preconditions.checkNotNull(actoolzipOutput);
    this.bundleContentArtifacts = Preconditions.checkNotNull(bundleContentArtifacts);
  }

  /**
   * The bundle root. This is the directory in the bundle zip archive in which every file is found
   * including the linked binary, nested bundles, and everything returned by
   * {@link #getExtraBundleFiles()}. In an application bundle, for instance, this function returns
   * {@code "Payload/(name).app"}.
   */
  public String getBundleRoot() {
    return String.format("Payload/%s.app", name);
  }

  /**
   * The name of the bundle, from which the bundle root and the path of the linked binary in the
   * bundle archive are derived.
   */
  public String getName() {
    return name;
  }

  /**
   * The artifact for the linked binary.
   */
  public Artifact getLinkedBinary() {
    return linkedBinary;
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
   * Returns the artifacts that are required to generate this bundle.
   */
  public NestedSet<Artifact> getBundleContentArtifacts() {
    return bundleContentArtifacts;
  }
}
