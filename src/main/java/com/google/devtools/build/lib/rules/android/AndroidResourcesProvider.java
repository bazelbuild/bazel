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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;

/** A provider that supplies ResourceContainers from its transitive closure. */
@AutoValue
@Immutable
public abstract class AndroidResourcesProvider implements TransitiveInfoProvider {

  public static AndroidResourcesProvider create(
      Label label,
      NestedSet<ResourceContainer> transitiveAndroidResources,
      NestedSet<ResourceContainer> directAndroidResources) {
    return new AutoValue_AndroidResourcesProvider(
        label, transitiveAndroidResources, directAndroidResources);
  }

  /**
   * Returns the label that is associated with this piece of information.
   */
  public abstract Label getLabel();

  /** Returns the transitive ResourceContainers for the label. */
  public abstract NestedSet<ResourceContainer> getTransitiveAndroidResources();

  /** Returns the immediate ResourceContainers for the label. */
  public abstract NestedSet<ResourceContainer> getDirectAndroidResources();


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
  @AutoValue
  @Immutable
  public abstract static class ResourceContainer {

    public static ResourceContainer create(
        Label label,
        @Nullable String javaPackage,
        @Nullable String renameManifestPackage,
        boolean constantsInlined,
        @Nullable Artifact apk,
        Artifact manifest,
        @Nullable Artifact javaSourceJar,
        @Nullable Artifact javaClassJar,
        ImmutableList<Artifact> assets,
        ImmutableList<Artifact> resources,
        ImmutableList<PathFragment> assetsRoots,
        ImmutableList<PathFragment> resourcesRoots,
        boolean manifestExported,
        @Nullable Artifact rTxt,
        @Nullable Artifact symbolsTxt) {
      return new AutoValue_AndroidResourcesProvider_ResourceContainer(
          label,
          javaPackage,
          renameManifestPackage,
          constantsInlined,
          apk,
          manifest,
          javaSourceJar,
          javaClassJar,
          assets,
          resources,
          assetsRoots,
          resourcesRoots,
          manifestExported,
          rTxt,
          symbolsTxt);
    }

    public abstract Label getLabel();
    @Nullable public abstract String getJavaPackage();
    @Nullable public abstract String getRenameManifestPackage();
    public abstract boolean getConstantsInlined();
    @Nullable public abstract Artifact getApk();
    public abstract Artifact getManifest();
    @Nullable public abstract Artifact getJavaSourceJar();
    @Nullable public abstract Artifact getJavaClassJar();

    abstract ImmutableList<Artifact> getAssets();
    abstract ImmutableList<Artifact> getResources();

    public ImmutableList<Artifact> getArtifacts(ResourceType resourceType) {
      return resourceType == ResourceType.ASSETS ? getAssets() : getResources();
    }

    public Iterable<Artifact> getArtifacts() {
      return Iterables.concat(getAssets(), getResources());
    }

    abstract ImmutableList<PathFragment> getAssetsRoots();
    abstract ImmutableList<PathFragment> getResourcesRoots();
    public ImmutableList<PathFragment> getRoots(ResourceType resourceType) {
      return resourceType == ResourceType.ASSETS ? getAssetsRoots() : getResourcesRoots();
    }

    public abstract boolean isManifestExported();
    @Nullable public abstract Artifact getRTxt();
    @Nullable public abstract Artifact getSymbolsTxt();

    // TODO(somebody) evaluate if we can just use hashCode and equals from AutoValue
    @Override
    public int hashCode() {
      return Objects.hash(getLabel(), getRTxt(), getSymbolsTxt());
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
      return Objects.equals(getLabel(), other.getLabel())
          && Objects.equals(getRTxt(), other.getRTxt())
          && Objects.equals(getSymbolsTxt(), other.getSymbolsTxt());
    }
  }

  AndroidResourcesProvider() {}
}
