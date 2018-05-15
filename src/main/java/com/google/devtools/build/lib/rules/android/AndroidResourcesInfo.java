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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** A provider that supplies ResourceContainers from its transitive closure. */
@SkylarkModule(
    name = "AndroidResourcesInfo",
    doc = "Android resources provided by a rule",
    category = SkylarkModuleCategory.PROVIDER)
@Immutable
public class AndroidResourcesInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "AndroidResourcesInfo";
  public static final NativeProvider<AndroidResourcesInfo> PROVIDER =
      new NativeProvider<AndroidResourcesInfo>(AndroidResourcesInfo.class, SKYLARK_NAME) {};

  /*
   * Local information about the target that produced this provider, for tooling. These values will
   * be made available even if they should not be inherited (for example, if this target has
   * "neverlink" set) - do not inherit them directly.
   */

  // Lets us know where the provider came from
  private final Label label;

  // An updated manifest - resource processing sometimes does additional manifest processing
  // TODO(b/30817309): Remove this once resource processing no longer does manifest processing
  private final ProcessedAndroidManifest manifest;

  // An R.txt file containing a list of all transitive resources this target expected
  private final Artifact rTxt;

  /*
   * Transitive information used for resource processing
   */

  private final NestedSet<ValidatedAndroidData> transitiveAndroidResources;
  private final NestedSet<ValidatedAndroidData> directAndroidResources;
  private final NestedSet<Artifact> transitiveResources;
  private final NestedSet<Artifact> transitiveAssets;
  private final NestedSet<Artifact> transitiveManifests;
  private final NestedSet<Artifact> transitiveAapt2RTxt;
  private final NestedSet<Artifact> transitiveSymbolsBin;
  private final NestedSet<Artifact> transitiveCompiledSymbols;
  private final NestedSet<Artifact> transitiveStaticLib;
  private final NestedSet<Artifact> transitiveRTxt;

  AndroidResourcesInfo(
      Label label,
      ProcessedAndroidManifest manifest,
      Artifact rTxt,
      NestedSet<ValidatedAndroidData> transitiveAndroidResources,
      NestedSet<ValidatedAndroidData> directAndroidResources,
      NestedSet<Artifact> transitiveResources,
      NestedSet<Artifact> transitiveAssets,
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
    this.transitiveAssets = transitiveAssets;
    this.transitiveManifests = transitiveManifests;
    this.transitiveAapt2RTxt = transitiveAapt2RTxt;
    this.transitiveSymbolsBin = transitiveSymbolsBin;
    this.transitiveCompiledSymbols = transitiveCompiledSymbols;
    this.transitiveStaticLib = transitiveStaticLib;
    this.transitiveRTxt = transitiveRTxt;
  }

  /** Returns the label that is associated with this piece of information. */
  @SkylarkCallable(name = "label", doc = "Returns the label for this target.", structField = true)
  public Label getLabel() {
    return label;
  }

  public ProcessedAndroidManifest getManifest() {
    return manifest;
  }

  public Artifact getRTxt() {
    return rTxt;
  }

  /** Returns the transitive ResourceContainers for the label. */
  @SkylarkCallable(
      name = "transitive_android_resources",
      doc = "Returns the transitive android resources for the label.",
      structField = true)
  public NestedSet<ValidatedAndroidData> getTransitiveAndroidResources() {
    return transitiveAndroidResources;
  }

  /** Returns the immediate ResourceContainers for the label. */
  @SkylarkCallable(
      name = "direct_android_resources",
      doc = "Returns the immediate android resources for the label.",
      structField = true)
  public NestedSet<ValidatedAndroidData> getDirectAndroidResources() {
    return directAndroidResources;
  }

  public NestedSet<Artifact> getTransitiveResources() {
    return transitiveResources;
  }

  /** @deprecated Assets are being decoupled from resources */
  @Deprecated
  public NestedSet<Artifact> getTransitiveAssets() {
    return transitiveAssets;
  }

  public NestedSet<Artifact> getTransitiveManifests() {
    return transitiveManifests;
  }

  public NestedSet<Artifact> getTransitiveAapt2RTxt() {
    return transitiveAapt2RTxt;
  }

  public NestedSet<Artifact> getTransitiveSymbolsBin() {
    return transitiveSymbolsBin;
  }

  public NestedSet<Artifact> getTransitiveCompiledSymbols() {
    return transitiveCompiledSymbols;
  }

  public NestedSet<Artifact> getTransitiveStaticLib() {
    return transitiveStaticLib;
  }

  public NestedSet<Artifact> getTransitiveRTxt() {
    return transitiveRTxt;
  }
}
