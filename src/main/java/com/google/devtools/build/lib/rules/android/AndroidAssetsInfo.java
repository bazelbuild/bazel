// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import javax.annotation.Nullable;

/** Provides information about transitive Android assets. */
@SkylarkModule(
    name = "AndroidAssetsInfo",
    doc = "Information about the Android assets provided by a rule.",
    category = SkylarkModuleCategory.PROVIDER)
public class AndroidAssetsInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "AndroidAssetsInfo";

  public static final NativeProvider<AndroidAssetsInfo> PROVIDER =
      new NativeProvider<AndroidAssetsInfo>(AndroidAssetsInfo.class, SKYLARK_NAME) {};

  private final Label label;
  @Nullable private final Artifact validationResult;
  private final NestedSet<ParsedAndroidAssets> directParsedAssets;
  private final NestedSet<ParsedAndroidAssets> transitiveParsedAssets;
  private final NestedSet<Artifact> transitiveAssets;
  private final NestedSet<Artifact> transitiveSymbols;

  static AndroidAssetsInfo empty(Label label) {
    return new AndroidAssetsInfo(
        label,
        null,
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));
  }

  public static AndroidAssetsInfo of(
      Label label,
      @Nullable Artifact validationResult,
      NestedSet<ParsedAndroidAssets> directParsedAssets,
      NestedSet<ParsedAndroidAssets> transitiveParsedAssets,
      NestedSet<Artifact> transitiveAssets,
      NestedSet<Artifact> transitiveSymbols) {
    return new AndroidAssetsInfo(
        label,
        validationResult,
        directParsedAssets,
        transitiveParsedAssets,
        transitiveAssets,
        transitiveSymbols);
  }

  private AndroidAssetsInfo(
      Label label,
      @Nullable Artifact validationResult,
      NestedSet<ParsedAndroidAssets> directParsedAssets,
      NestedSet<ParsedAndroidAssets> transitiveParsedAssets,
      NestedSet<Artifact> transitiveAssets,
      NestedSet<Artifact> transitiveSymbols) {
    super(PROVIDER);
    this.label = label;
    this.validationResult = validationResult;
    this.directParsedAssets = directParsedAssets;
    this.transitiveParsedAssets = transitiveParsedAssets;
    this.transitiveAssets = transitiveAssets;
    this.transitiveSymbols = transitiveSymbols;
  }

  public Label getLabel() {
    return label;
  }

  @SkylarkCallable(
      name = "validation_result",
      structField = true,
      allowReturnNones = true,
      doc =
          "If not None, represents the output of asset merging and validation for this target. The"
              + " action to merge and validate assets is not run be default; to force it, add this"
              + " artifact to your target's outputs. The validation action is somewhat expensive -"
              + " in native code, this artifact is added to the top-level output group (so"
              + " validation is only done if the target is requested on the command line). The"
              + " contents of this artifact are subject to change and should not be relied upon.")
  @Nullable
  public Artifact getValidationResult() {
    return validationResult;
  }

  public NestedSet<ParsedAndroidAssets> getDirectParsedAssets() {
    return directParsedAssets;
  }

  public NestedSet<ParsedAndroidAssets> getTransitiveParsedAssets() {
    return transitiveParsedAssets;
  }

  public NestedSet<Artifact> getAssets() {
    return transitiveAssets;
  }

  public NestedSet<Artifact> getSymbols() {
    return transitiveSymbols;
  }
}
