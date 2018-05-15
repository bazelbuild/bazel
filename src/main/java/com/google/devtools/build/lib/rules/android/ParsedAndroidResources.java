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
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** Wraps parsed (and, if requested, compiled) android resources. */
public class ParsedAndroidResources extends AndroidResources
    implements CompiledMergableAndroidData {
  private final Artifact symbols;
  @Nullable private final Artifact compiledSymbols;
  private final Label label;
  private final StampedAndroidManifest manifest;

  public static ParsedAndroidResources parseFrom(
      RuleContext ruleContext,
      AndroidResources resources,
      StampedAndroidManifest manifest,
      boolean enableDataBinding,
      AndroidAaptVersion aaptVersion)
      throws InterruptedException {

    boolean isAapt2 = aaptVersion == AndroidAaptVersion.AAPT2;

    AndroidResourceParsingActionBuilder builder =
        new AndroidResourceParsingActionBuilder(ruleContext);

    if (enableDataBinding && isAapt2) {
      // TODO(corysmith): Centralize the data binding processing and zipping into a single
      // action. Data binding processing needs to be triggered here as well as the merger to
      // avoid aapt2 from throwing an error during compilation.
      builder.setDataBindingInfoZip(DataBinding.getSuffixedInfoFile(ruleContext, "_unused"));
    }

    return builder
        .setOutput(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_MERGED_SYMBOLS))
        .setCompiledSymbolsOutput(
            isAapt2
                ? ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_COMPILED_SYMBOLS)
                : null)
        .build(resources, manifest);
  }

  public static ParsedAndroidResources of(
      AndroidResources resources,
      Artifact symbols,
      @Nullable Artifact compiledSymbols,
      Label label,
      StampedAndroidManifest manifest) {
    return new ParsedAndroidResources(resources, symbols, compiledSymbols, label, manifest);
  }

  ParsedAndroidResources(ParsedAndroidResources other, StampedAndroidManifest manifest) {
    this(other, other.symbols, other.compiledSymbols, other.label, manifest);
  }

  private ParsedAndroidResources(
      AndroidResources resources,
      Artifact symbols,
      @Nullable Artifact compiledSymbols,
      Label label,
      StampedAndroidManifest manifest) {
    super(resources);
    this.symbols = symbols;
    this.compiledSymbols = compiledSymbols;
    this.label = label;
    this.manifest = manifest;
  }

  @Override
  public Artifact getSymbols() {
    return symbols;
  }

  @Override
  @Nullable
  public Artifact getCompiledSymbols() {
    return compiledSymbols;
  }

  @Override
  public Iterable<Artifact> getArtifacts() {
    return getResources();
  }

  @Override
  public Artifact getManifest() {
    return manifest.getManifest();
  }

  @Override
  public boolean isManifestExported() {
    return manifest.isExported();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  public String getJavaPackage() {
    return manifest.getPackage();
  }

  public StampedAndroidManifest getStampedManifest() {
    return manifest;
  }

  /** Merges this target's resources with resources from dependencies. */
  MergedAndroidResources merge(
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      boolean enableDataBinding,
      AndroidAaptVersion aaptVersion)
      throws InterruptedException {
    return MergedAndroidResources.mergeFrom(
        ruleContext, this, resourceDeps, enableDataBinding, aaptVersion);
  }

  @Override
  public Optional<? extends ParsedAndroidResources> maybeFilter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return super.maybeFilter(errorConsumer, resourceFilter, isDependency)
        .map(
            resources ->
                ParsedAndroidResources.of(resources, symbols, compiledSymbols, label, manifest));
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object)) {
      return false;
    }

    ParsedAndroidResources other = (ParsedAndroidResources) object;
    return symbols.equals(other.symbols)
        && Objects.equals(compiledSymbols, other.compiledSymbols)
        && label.equals(other.label)
        && manifest.equals(other.manifest);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), symbols, compiledSymbols, label, manifest);
  }
}
