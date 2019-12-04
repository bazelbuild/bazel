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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingContext;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** Wraps parsed (and, if requested, compiled) android resources. */
public class ParsedAndroidResources extends AndroidResources {
  private final Artifact symbols;
  @Nullable private final Artifact compiledSymbols;
  private final Label label;
  private final StampedAndroidManifest manifest;
  private final DataBindingContext dataBindingContext;

  public static ParsedAndroidResources parseFrom(
      AndroidDataContext dataContext,
      AndroidResources resources,
      StampedAndroidManifest manifest,
      AndroidAaptVersion aaptVersion,
      DataBindingContext dataBindingContext)
      throws InterruptedException {

    boolean isAapt2 = aaptVersion == AndroidAaptVersion.AAPT2;
    Preconditions.checkState(isAapt2);

    AndroidResourceParsingActionBuilder builder = new AndroidResourceParsingActionBuilder();

    // TODO(b/120093531): This is only used in Databinding v1.
    dataBindingContext.supplyLayoutInfo(builder::setDataBindingInfoZip);
    // In databinding v2, this strips out the databinding and generates the layout info file.
    AndroidResources databindingProcessedResources =
        dataBindingContext.processResources(dataContext, resources, manifest.getPackage());

    return builder
        .setOutput(dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_MERGED_SYMBOLS))
        .setCompiledSymbolsOutput(
            isAapt2
                ? dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_COMPILED_SYMBOLS)
                : null)
        .build(
            dataContext,
            databindingProcessedResources,
            manifest,
            dataBindingContext);
  }

  @VisibleForTesting
  static Artifact getDummyDataBindingArtifact(ActionConstructionContext dataContext) {
    return dataContext.getUniqueDirectoryArtifact("dummydatabinding", "unused.zip");
  }

  public static ParsedAndroidResources of(
      AndroidResources resources,
      Artifact symbols,
      @Nullable Artifact compiledSymbols,
      Label label,
      StampedAndroidManifest manifest,
      DataBindingContext dataBindingContext) {
    return new ParsedAndroidResources(
        resources, symbols, compiledSymbols, label, manifest, dataBindingContext);
  }

  ParsedAndroidResources(ParsedAndroidResources other, StampedAndroidManifest manifest) {
    this(
        other,
        other.symbols,
        other.compiledSymbols,
        other.label,
        manifest,
        other.dataBindingContext);
  }

  protected ParsedAndroidResources(
      AndroidResources resources,
      Artifact symbols,
      @Nullable Artifact compiledSymbols,
      Label label,
      StampedAndroidManifest manifest,
      DataBindingContext dataBindingContext) {
    super(resources);
    this.symbols = symbols;
    this.compiledSymbols = compiledSymbols;
    this.label = label;
    this.manifest = manifest;
    this.dataBindingContext = dataBindingContext;
  }

  public Artifact getSymbols() {
    return symbols;
  }

  @Nullable
  public Artifact getCompiledSymbols() {
    return compiledSymbols;
  }

  public Iterable<Artifact> getArtifacts() {
    return getResources();
  }

  public Artifact getManifest() {
    return manifest.getManifest();
  }

  public boolean isManifestExported() {
    return manifest.isExported();
  }

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
      AndroidDataContext dataContext,
      ResourceDependencies resourceDeps,
      AndroidAaptVersion aaptVersion)
      throws InterruptedException {
    return MergedAndroidResources.mergeFrom(dataContext, this, resourceDeps, aaptVersion);
  }

  @Override
  public Optional<? extends ParsedAndroidResources> maybeFilter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return super.maybeFilter(errorConsumer, resourceFilter, isDependency)
        .map(
            resources ->
                ParsedAndroidResources.of(
                    resources, symbols, compiledSymbols, label, manifest, dataBindingContext));
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object) || !(object instanceof ParsedAndroidResources)) {
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

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("symbols", symbols)
        .add("compiledSymbols", compiledSymbols)
        .add("label", label)
        .add("manifest", manifest)
        .toString();
  }

  public DataBindingContext asDataBindingContext() {
    return dataBindingContext;
  }
}
