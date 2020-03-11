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
import com.google.devtools.build.lib.skylarkbuildapi.android.ParsedAndroidAssetsApi;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.util.Objects;
import javax.annotation.Nullable;

/** Parsed Android assets which can be merged together with assets from dependencies. */
public class ParsedAndroidAssets extends AndroidAssets implements ParsedAndroidAssetsApi {

  public static final SkylarkType TYPE = SkylarkType.of(ParsedAndroidAssets.class);

  private final Artifact symbols;
  @Nullable private final Artifact compiledSymbols;
  private final Label label;

  public static ParsedAndroidAssets parseFrom(AndroidDataContext dataContext, AndroidAssets assets)
      throws InterruptedException {
    AndroidResourceParsingActionBuilder builder =
        new AndroidResourceParsingActionBuilder()
            .setOutput(dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_ASSET_SYMBOLS))
            .setCompiledSymbolsOutput(
                dataContext.createOutputArtifact(
                    AndroidRuleClasses.ANDROID_ASSET_COMPILED_SYMBOLS));

    return builder.build(dataContext, assets);
  }

  public static ParsedAndroidAssets of(
      AndroidAssets assets, Artifact symbols, @Nullable Artifact compiledSymbols, Label label) {
    return new ParsedAndroidAssets(assets, symbols, compiledSymbols, label);
  }

  ParsedAndroidAssets(ParsedAndroidAssets other) {
    this(other, other.symbols, other.compiledSymbols, other.label);
  }

  private ParsedAndroidAssets(
      AndroidAssets other, Artifact symbols, @Nullable Artifact compiledSymbols, Label label) {
    super(other);
    this.symbols = symbols;
    this.compiledSymbols = compiledSymbols;
    this.label = label;
  }

  MergedAndroidAssets merge(AndroidDataContext dataContext, AssetDependencies assetDeps)
      throws InterruptedException {
    return MergedAndroidAssets.mergeFrom(dataContext, this, assetDeps);
  }

  public Label getLabel() {
    return label;
  }

  public Artifact getSymbols() {
    return symbols;
  }

  @Nullable
  public Artifact getCompiledSymbols() {
    return compiledSymbols;
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
