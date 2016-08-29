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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A provider that supplies ResourceContainers from its transitive closure.
 */
@Immutable
public final class AndroidResourcesProvider implements TransitiveInfoProvider {
  private final Label label;
  private final NestedSet<ResourceContainer> transitiveAndroidResources;
  private final NestedSet<ResourceContainer> directAndroidResources;

  public AndroidResourcesProvider(
      Label label, NestedSet<ResourceContainer> transitiveAndroidResources,
      NestedSet<ResourceContainer> directAndroidResources) {
    this.label = label;
    this.directAndroidResources = directAndroidResources;
    this.transitiveAndroidResources = transitiveAndroidResources;
  }

  /**
   * Returns the label that is associated with this piece of information.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns the transitive ResourceContainers for the label.
   */
  public NestedSet<ResourceContainer> getTransitiveAndroidResources() {
    return transitiveAndroidResources;
  }

  /**
   * Returns the immediate ResourceContainers for the label.
   */
  public NestedSet<ResourceContainer> getDirectAndroidResources() {
    return directAndroidResources;
  }


  /**
   * The type of resource in question: either asset or a resource.
   */
  public enum ResourceType {
    ASSETS("assets"),
    RESOURCES("resources");

    private final String attribute;

    private ResourceType(String attribute) {
      this.attribute = attribute;
    }

    public String getAttribute() {
      return attribute;
    }
  }


  /**
   * The resources contributed by a single target.
   */
  @Immutable
  public static final class ResourceContainer {
    private final Label label;
    private final String javaPackage;
    private final String renameManifestPackage;
    private final boolean constantsInlined;
    private final Artifact apk;
    private final Artifact manifest;
    private final ImmutableList<Artifact> assets;
    private final ImmutableList<Artifact> resources;
    private final ImmutableList<PathFragment> assetsRoots;
    private final ImmutableList<PathFragment> resourcesRoots;
    private final boolean manifestExported;
    private final Artifact javaSourceJar;
    private final Artifact javaClassJar;
    private final Artifact rTxt;
    private final Artifact symbolsTxt;

    public ResourceContainer(Label label,
        String javaPackage,
        @Nullable String renameManifestPackage,
        boolean constantsInlined,
        @Nullable Artifact apk,
        Artifact manifest,
        Artifact javaSourceJar,
        @Nullable Artifact javaClassJar,
        ImmutableList<Artifact> assets,
        ImmutableList<Artifact> resources,
        ImmutableList<PathFragment> assetsRoots,
        ImmutableList<PathFragment> resourcesRoots,
        boolean manifestExported,
        Artifact rTxt,
        Artifact symbolsTxt) {
      this.javaSourceJar = javaSourceJar;
      this.javaClassJar = javaClassJar;
      this.manifestExported = manifestExported;
      this.label = Preconditions.checkNotNull(label);
      this.javaPackage = Preconditions.checkNotNull(javaPackage);
      this.renameManifestPackage = renameManifestPackage;
      this.constantsInlined = constantsInlined;
      this.apk = apk;
      this.manifest = Preconditions.checkNotNull(manifest);
      this.assets = Preconditions.checkNotNull(assets);
      this.resources = Preconditions.checkNotNull(resources);
      this.assetsRoots = Preconditions.checkNotNull(assetsRoots);
      this.resourcesRoots = Preconditions.checkNotNull(resourcesRoots);
      this.rTxt = rTxt;
      this.symbolsTxt = symbolsTxt;
    }

    public Label getLabel() {
      return label;
    }

    public String getJavaPackage() {
      return javaPackage;
    }

    public String getRenameManifestPackage() {
      return renameManifestPackage;
    }

    public boolean getConstantsInlined() {
      return constantsInlined;
    }

    public Artifact getApk() {
      return apk;
    }

    public Artifact getJavaSourceJar() {
      return javaSourceJar;
    }

    public Artifact getJavaClassJar() {
      return javaClassJar;
    }

    public Artifact getManifest() {
      return manifest;
    }

    public boolean isManifestExported() {
      return manifestExported;
    }

    public ImmutableList<Artifact> getArtifacts(ResourceType resourceType) {
      return resourceType == ResourceType.ASSETS ? assets : resources;
    }

    public Iterable<Artifact> getArtifacts() {
      return Iterables.concat(assets, resources);
    }

    public Artifact getRTxt() {
      return rTxt;
    }


    public Artifact getSymbolsTxt() {
      return symbolsTxt;
    }

    public ImmutableList<PathFragment> getRoots(ResourceType resourceType) {
      return resourceType == ResourceType.ASSETS ? assetsRoots : resourcesRoots;
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, rTxt, symbolsTxt);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof ResourceContainer)) {
        return false;
      }
      ResourceContainer other = (ResourceContainer) obj;
      return Objects.equals(label, other.label)
          && Objects.equals(rTxt, other.rTxt)
          && Objects.equals(symbolsTxt, other.symbolsTxt);
    }

    @Override
    public String toString() {
      return String.format(
          "ResourceContainer [label=%s, javaPackage=%s, renameManifestPackage=%s,"
          + " constantsInlined=%s, apk=%s, manifest=%s, assets=%s, resources=%s, assetsRoots=%s,"
          + " resourcesRoots=%s, manifestExported=%s, javaSourceJar=%s, javaClassJar=%s,"
          + " rTxt=%s, symbolsTxt=%s]",
          label, javaPackage, renameManifestPackage, constantsInlined, apk, manifest, assets,
          resources, assetsRoots, resourcesRoots, manifestExported, javaSourceJar,
          javaClassJar, rTxt, symbolsTxt);
    }
  }
}
