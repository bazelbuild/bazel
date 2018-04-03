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
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import java.util.Objects;
import javax.annotation.Nullable;

/** Wraps parsed (and, if requested, compiled) android resources. */
public class ParsedAndroidResources extends AndroidResources {
  private final Artifact symbols;
  @Nullable private final Artifact compiledSymbols;
  private final Label label;

  public static ParsedAndroidResources parseFrom(
      RuleContext ruleContext, AndroidResources resources, StampedAndroidManifest manifest)
      throws RuleErrorException, InterruptedException {

    boolean isAapt2 =
        AndroidAaptVersion.chooseTargetAaptVersion(ruleContext).equals(AndroidAaptVersion.AAPT2);

    AndroidResourceParsingActionBuilder builder =
        new AndroidResourceParsingActionBuilder(ruleContext);

    if (DataBinding.isEnabled(ruleContext) && isAapt2) {
      // TODO(corysmith): Centralize the data binding processing and zipping into a single
      // action. Data binding processing needs to be triggered here as well as the merger to
      // avoid aapt2 from throwing an error during compilation.
      builder
          .setDataBindingInfoZip(DataBinding.getSuffixedInfoFile(ruleContext, "_unused"))
          .setManifest(manifest.getManifest())
          .setJavaPackage(manifest.getPackage());
    }

    return builder
        .setOutput(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_MERGED_SYMBOLS))
        .setCompiledSymbolsOutput(
            isAapt2
                ? ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_COMPILED_SYMBOLS)
                : null)
        .build(resources);
  }

  public static ParsedAndroidResources of(
      AndroidResources resources,
      Artifact symbols,
      @Nullable Artifact compiledSymbols,
      Label label) {
    return new ParsedAndroidResources(
        resources,
        symbols,
        compiledSymbols,
        label);
  }

  ParsedAndroidResources(ParsedAndroidResources other) {
    this(other, other.symbols, other.compiledSymbols, other.label);
  }

  private ParsedAndroidResources(
      AndroidResources resources,
      Artifact symbols,
      @Nullable Artifact compiledSymbols,
      Label label) {
    super(resources);
    this.symbols = symbols;
    this.compiledSymbols = compiledSymbols;
    this.label = label;
  }

  public Artifact getSymbols() {
    return symbols;
  }

  @Nullable
  public Artifact getCompiledSymbols() {
    return compiledSymbols;
  }

  public Label getLabel() {
    return label;
  }

  @Override
  public boolean equals(Object object) {
    if (!(object instanceof ParsedAndroidResources) || !super.equals(object)) {
      return false;
    }

    ParsedAndroidResources other = (ParsedAndroidResources) object;
    return symbols.equals(other.symbols)
        && Objects.equals(compiledSymbols, other.compiledSymbols)
        && label.equals(other.label);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), symbols, compiledSymbols, label);
  }
}
