// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A provider that supplies Android resources from its transitive closure.
 */
@Immutable
public final class AndroidResourcesProvider implements TransitiveInfoProvider {

  private final Label label;
  private final NestedSet<ResourceContainer> transitiveAndroidResources;

  public AndroidResourcesProvider(Label label,
      NestedSet<ResourceContainer> transitiveAndroidResources) {
    this.label = label;
    this.transitiveAndroidResources = transitiveAndroidResources;
  }

  /**
   * Returns the label that is associated with this piece of information.
   *
   * <p>
   * This is usually the label of the target that provides the information.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns transitive Android resources (APK, assets, etc.).
   */
  public NestedSet<ResourceContainer> getTransitiveAndroidResources() {
    return transitiveAndroidResources;
  }


  /**
   * The type of resource in question: either asset or a resource.
   */
  public enum ResourceType {
    ASSETS("assets"), RESOURCES("resources");

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
    private final Artifact rTxt;

    public ResourceContainer(Label label,
        String javaPackage,
        @Nullable String renameManifestPackage,
        boolean constantsInlined,
        Artifact apk,
        Artifact manifest,
        Artifact javaSourceJar,
        ImmutableList<Artifact> assets,
        ImmutableList<Artifact> resources,
        ImmutableList<PathFragment> assetsRoots,
        ImmutableList<PathFragment> resourcesRoots,
        boolean manifestExported,
        Artifact rTxt) {
      this.javaSourceJar = javaSourceJar;
      this.manifestExported = manifestExported;
      this.label = Preconditions.checkNotNull(label);
      this.javaPackage = Preconditions.checkNotNull(javaPackage);
      this.renameManifestPackage = renameManifestPackage;
      this.constantsInlined = constantsInlined;
      this.apk = Preconditions.checkNotNull(apk);
      this.manifest = Preconditions.checkNotNull(manifest);
      this.assets = Preconditions.checkNotNull(assets);
      this.resources = Preconditions.checkNotNull(resources);
      this.assetsRoots = Preconditions.checkNotNull(assetsRoots);
      this.resourcesRoots = Preconditions.checkNotNull(resourcesRoots);
      this.rTxt = rTxt;
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

    public ImmutableList<PathFragment> getRoots(ResourceType resourceType) {
      return resourceType == ResourceType.ASSETS ? assetsRoots : resourcesRoots;
    }

    @Override
    public int hashCode() {
      return Objects.hash(label);
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
      return label.equals(other.label);
    }
  }
}
