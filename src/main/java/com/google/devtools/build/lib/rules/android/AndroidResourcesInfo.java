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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidResourcesInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

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
  public NestedSet<ValidatedAndroidResources> getTransitiveAndroidResources() {
    return transitiveAndroidResources;
  }

  @Override
  public NestedSet<ValidatedAndroidResources> getDirectAndroidResources() {
    return directAndroidResources;
  }

  @Override
  public NestedSet<Artifact> getTransitiveResources() {
    return transitiveResources;
  }

  @Override
  public NestedSet<Artifact> getTransitiveManifests() {
    return transitiveManifests;
  }

  @Override
  public NestedSet<Artifact> getTransitiveAapt2RTxt() {
    return transitiveAapt2RTxt;
  }

  @Override
  public NestedSet<Artifact> getTransitiveSymbolsBin() {
    return transitiveSymbolsBin;
  }

  @Override
  public NestedSet<Artifact> getTransitiveCompiledSymbols() {
    return transitiveCompiledSymbols;
  }

  @Override
  public NestedSet<Artifact> getTransitiveStaticLib() {
    return transitiveStaticLib;
  }

  @Override
  public NestedSet<Artifact> getTransitiveRTxt() {
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
        SkylarkNestedSet transitiveAndroidResources,
        SkylarkNestedSet directAndroidResources,
        SkylarkNestedSet transitiveResources,
        SkylarkNestedSet transitiveManifests,
        SkylarkNestedSet transitiveAapt2RTxt,
        SkylarkNestedSet transitiveSymbolsBin,
        SkylarkNestedSet transitiveCompiledSymbols,
        SkylarkNestedSet transitiveStaticLib,
        SkylarkNestedSet transitiveRTxt)
        throws EvalException {
      return new AndroidResourcesInfo(
          label,
          manifest,
          rTxt,
          nestedSet(transitiveAndroidResources, ValidatedAndroidResources.class),
          nestedSet(directAndroidResources, ValidatedAndroidResources.class),
          nestedSet(transitiveResources, Artifact.class),
          nestedSet(transitiveManifests, Artifact.class),
          nestedSet(transitiveAapt2RTxt, Artifact.class),
          nestedSet(transitiveSymbolsBin, Artifact.class),
          nestedSet(transitiveCompiledSymbols, Artifact.class),
          nestedSet(transitiveStaticLib, Artifact.class),
          nestedSet(transitiveRTxt, Artifact.class));
    }

    private <T> NestedSet<T> nestedSet(SkylarkNestedSet from, Class<T> with) {
      return NestedSetBuilder.<T>stableOrder().addTransitive(from.getSet(with)).build();
    }
  }
}
