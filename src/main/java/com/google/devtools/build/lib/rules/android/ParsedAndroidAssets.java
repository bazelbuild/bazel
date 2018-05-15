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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Objects;

/** Parsed Android assets which can be merged together with assets from dependencies. */
public class ParsedAndroidAssets extends AndroidAssets implements MergableAndroidData {
  private final Artifact symbols;
  private final Label label;

  public static ParsedAndroidAssets parseFrom(RuleContext ruleContext, AndroidAssets assets)
      throws InterruptedException {
    return new AndroidResourceParsingActionBuilder(ruleContext)
        .setOutput(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_ASSET_SYMBOLS))
        .build(assets);
  }

  public static ParsedAndroidAssets of(AndroidAssets assets, Artifact symbols, Label label) {
    return new ParsedAndroidAssets(assets, symbols, label);
  }

  ParsedAndroidAssets(ParsedAndroidAssets other) {
    this(other, other.symbols, other.label);
  }

  private ParsedAndroidAssets(AndroidAssets other, Artifact symbols, Label label) {
    super(other);
    this.symbols = symbols;
    this.label = label;
  }

  /** Merges these assets with assets from dependencies. */
  MergedAndroidAssets merge(RuleContext ruleContext, AssetDependencies assetDeps)
      throws InterruptedException {
    return MergedAndroidAssets.mergeFrom(ruleContext, this, assetDeps);
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public Artifact getSymbols() {
    return symbols;
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object)) {
      return false;
    }

    ParsedAndroidAssets other = (ParsedAndroidAssets) object;
    return symbols.equals(other.symbols) && label.equals(other.label);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), symbols, label);
  }
}
