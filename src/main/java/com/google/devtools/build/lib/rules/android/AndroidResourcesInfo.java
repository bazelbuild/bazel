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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidResourcesInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A provider that supplies ResourceContainers from its transitive closure. */
@Immutable
public class AndroidResourcesInfo extends NativeInfo
    implements AndroidResourcesInfoApi<Artifact, ValidatedAndroidResources, AndroidManifestInfo> {

  public static final String PROVIDER_NAME = "AndroidResourcesInfo";
  public static final Provider PROVIDER = new Provider();

  /*
   * Local information about the target that produced this provider, for tooling. These values will
   * be made available even if they should not be inherited (for example, if this target has
   * "neverlink" set) - do not inherit them directly.
   */

  // Lets us know where the provider came from
  private final Label label;

  // An updated manifest - resource processing sometimes does additional manifest processing
  // TODO(b/30817309): Remove this once resource processing no longer does manifest processing
  private final AndroidManifestInfo manifest;

  // An R.txt file containing a list of all transitive resources this target expected
  private final Artifact rTxt;

  /*
   * Transitive information used for resource processing
   */

  private final NestedSet<ValidatedAndroidResources> transitiveAndroidResources;
  private final NestedSet<ValidatedAndroidResources> directAndroidResources;
  private final NestedSet<Artifact> transitiveResources;
  private final NestedSet<Artifact> transitiveManifests;
  private final NestedSet<Artifact> transitiveAapt2RTxt;
  private final NestedSet<Artifact> transitiveAapt2ValidationArtifacts;
  private final NestedSet<Artifact> transitiveSymbolsBin;
  private final NestedSet<Artifact> transitiveCompiledSymbols;
  private final NestedSet<Artifact> transitiveStaticLib;
  private final NestedSet<Artifact> transitiveRTxt;

  AndroidResourcesInfo(
      Label label,
      AndroidManifestInfo manifest,
      Artifact rTxt,
      NestedSet<ValidatedAndroidResources> transitiveAndroidResources,
      NestedSet<ValidatedAndroidResources> directAndroidResources,
      NestedSet<Artifact> transitiveResources,
      NestedSet<Artifact> transitiveManifests,
      NestedSet<Artifact> transitiveAapt2RTxt,
      NestedSet<Artifact> transitiveAapt2ValidationArtifacts,
      NestedSet<Artifact> transitiveSymbolsBin,
      NestedSet<Artifact> transitiveCompiledSymbols,
      NestedSet<Artifact> transitiveStaticLib,
      NestedSet<Artifact> transitiveRTxt) {
    super(PROVIDER);
    this.label = label;
    this.manifest = manifest;
    this.rTxt = rTxt;
    this.transitiveAndroidResources = transitiveAndroidResources;
    this.directAndroidResources = directAndroidResources;
    this.transitiveResources = transitiveResources;
    this.transitiveManifests = transitiveManifests;
    this.transitiveAapt2RTxt = transitiveAapt2RTxt;
    this.transitiveAapt2ValidationArtifacts = transitiveAapt2ValidationArtifacts;
    this.transitiveSymbolsBin = transitiveSymbolsBin;
    this.transitiveCompiledSymbols = transitiveCompiledSymbols;
    this.transitiveStaticLib = transitiveStaticLib;
    this.transitiveRTxt = transitiveRTxt;
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public AndroidManifestInfo getManifest() {
    return manifest;
  }

  @Override
  public Artifact getRTxt() {
    return rTxt;
  }

  @Override
  public Depset /*<ValidatedAndroidResources>*/ getTransitiveAndroidResourcesForStarlark() {
    return Depset.of(ValidatedAndroidResources.TYPE, transitiveAndroidResources);
  }

  public NestedSet<ValidatedAndroidResources> getTransitiveAndroidResources() {
    return transitiveAndroidResources;
  }

  @Override
  public Depset /*<ValidatedAndroidResources>*/ getDirectAndroidResourcesForStarlark() {
    return Depset.of(ValidatedAndroidResources.TYPE, directAndroidResources);
  }

  public NestedSet<ValidatedAndroidResources> getDirectAndroidResources() {
    return directAndroidResources;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveResourcesForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveResources);
  }

  public NestedSet<Artifact> getTransitiveResources() {
    return transitiveResources;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveManifestsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveManifests);
  }

  public NestedSet<Artifact> getTransitiveManifests() {
    return transitiveManifests;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveAapt2RTxtForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveAapt2RTxt);
  }

  public NestedSet<Artifact> getTransitiveAapt2RTxt() {
    return transitiveAapt2RTxt;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveAapt2ValidationArtifactsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveAapt2ValidationArtifacts);
  }

  NestedSet<Artifact> getTransitiveAapt2ValidationArtifacts() {
    return transitiveAapt2ValidationArtifacts;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveSymbolsBinForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveSymbolsBin);
  }

  public NestedSet<Artifact> getTransitiveSymbolsBin() {
    return transitiveSymbolsBin;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveCompiledSymbolsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveCompiledSymbols);
  }

  NestedSet<Artifact> getTransitiveCompiledSymbols() {
    return transitiveCompiledSymbols;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveStaticLibForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveStaticLib);
  }

  NestedSet<Artifact> getTransitiveStaticLib() {
    return transitiveStaticLib;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveRTxtForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveRTxt);
  }

  NestedSet<Artifact> getTransitiveRTxt() {
    return transitiveRTxt;
  }

  /** Provider for {@link AndroidResourcesInfo}. */
  public static class Provider extends BuiltinProvider<AndroidResourcesInfo>
      implements AndroidResourcesInfoApi.AndroidResourcesInfoApiProvider<
          Artifact, ValidatedAndroidResources, AndroidManifestInfo> {

    private Provider() {
      super(PROVIDER_NAME, AndroidResourcesInfo.class);
    }

    @Override
    public AndroidResourcesInfo createInfo(
        Label label,
        AndroidManifestInfo manifest,
        Artifact rTxt,
        Depset transitiveAndroidResources,
        Depset directAndroidResources,
        Depset transitiveResources,
        Depset transitiveManifests,
        Depset transitiveAapt2RTxt,
        Depset transitiveSymbolsBin,
        Depset transitiveCompiledSymbols,
        Object transitiveStaticLib,
        Object transitiveRTxt,
        Object transitiveAapt2ValidationArtifacts)
        throws EvalException {
      return new AndroidResourcesInfo(
          label,
          manifest,
          rTxt,
          nestedSet(
              transitiveAndroidResources,
              ValidatedAndroidResources.class,
              "transitive_android_resources"),
          nestedSet(
              directAndroidResources, ValidatedAndroidResources.class, "direct_android_resources"),
          nestedSet(transitiveResources, Artifact.class, "transitive_resources"),
          nestedSet(transitiveManifests, Artifact.class, "transitive_manifests"),
          nestedSet(transitiveAapt2RTxt, Artifact.class, "transitive_aapt2_r_txt"),
          nestedSet(transitiveAapt2ValidationArtifacts, Artifact.class, "validation_artifacts"),
          nestedSet(transitiveSymbolsBin, Artifact.class, "transitive_symbols_bin"),
          nestedSet(transitiveCompiledSymbols, Artifact.class, "transitive_compiled_symbols"),
          nestedSet(transitiveStaticLib, Artifact.class, "transitive_static_lib"),
          nestedSet(transitiveRTxt, Artifact.class, "transitive_r_txt"));
    }

    private static <T> NestedSet<T> nestedSet(Depset from, Class<T> with, String fieldName)
        throws EvalException {
      return NestedSetBuilder.<T>stableOrder()
          .addTransitive(Depset.cast(from, with, fieldName))
          .build();
    }

    private static <T> NestedSet<T> nestedSet(Object from, Class<T> with, String fieldName)
        throws EvalException {
      Preconditions.checkArgument(
          from instanceof Depset || from == com.google.devtools.build.lib.syntax.Starlark.UNBOUND);

      if (from instanceof Depset) {
        return nestedSet((Depset) from, with, fieldName);
      }
      return NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
  }
}
