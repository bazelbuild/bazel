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
package com.google.devtools.build.lib.rules.android.databinding;

import static com.google.devtools.build.lib.rules.android.AndroidStarlarkData.fromNoneable;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.DataBindingV2ProviderApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/**
 * A provider that exposes this enables <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>
 * version 2 on its resource processing and Java compilation.
 */
public final class DataBindingV2Provider extends NativeInfo
    implements DataBindingV2ProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final NestedSet<Artifact> classInfos;

  private final NestedSet<Artifact> setterStores;

  private final NestedSet<Artifact> transitiveBRFiles;

  /** The label and java package of this rule and any rules that this rule exports. */
  private final ImmutableList<LabelJavaPackagePair> labelAndJavaPackages;

  private final NestedSet<LabelJavaPackagePair> transitiveLabelAndJavaPackages;

  public DataBindingV2Provider(
      NestedSet<Artifact> classInfos,
      NestedSet<Artifact> setterStores,
      NestedSet<Artifact> transitiveBRFiles,
      ImmutableList<LabelJavaPackagePair> labelAndJavaPackages,
      NestedSet<LabelJavaPackagePair> transitiveLabelAndJavaPackages) {
    this.classInfos = classInfos;
    this.setterStores = setterStores;
    this.transitiveBRFiles = transitiveBRFiles;
    this.labelAndJavaPackages = labelAndJavaPackages;
    this.transitiveLabelAndJavaPackages = transitiveLabelAndJavaPackages;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public Depset /*<Artifact>*/ getClassInfosForStarlark() {
    return Depset.of(Artifact.TYPE, classInfos);
  }

  public NestedSet<Artifact> getClassInfos() {
    return classInfos;
  }

  @Override
  public Depset /*<Artifact>*/ getSetterStoresForStarlark() {
    return Depset.of(Artifact.TYPE, setterStores);
  }

  public NestedSet<Artifact> getSetterStores() {
    return setterStores;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveBRFilesForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveBRFiles);
  }

  public NestedSet<Artifact> getTransitiveBRFiles() {
    return transitiveBRFiles;
  }

  @Override
  public ImmutableList<LabelJavaPackagePair> getLabelAndJavaPackages() {
    return labelAndJavaPackages;
  }

  @Override
  public Depset /*<LabelJavaPackagePair>*/ getTransitiveLabelAndJavaPackagesForStarlark() {
    return Depset.of(LabelJavaPackagePair.TYPE, transitiveLabelAndJavaPackages);
  }

  public NestedSet<LabelJavaPackagePair> getTransitiveLabelAndJavaPackages() {
    return transitiveLabelAndJavaPackages;
  }

  public static DataBindingV2Provider createProvider(
      Artifact setterStoreFile,
      Artifact classInfoFile,
      Artifact brFile,
      String label,
      String javaPackage,
      // ugh these *Api types do not help one bit
      Iterable<? extends DataBindingV2ProviderApi<Artifact>> databindingV2ProvidersInDeps,
      Iterable<? extends DataBindingV2ProviderApi<Artifact>> databindingV2ProvidersInExports) {

    NestedSetBuilder<Artifact> setterStoreFiles = NestedSetBuilder.stableOrder();
    if (setterStoreFile != null) {
      setterStoreFiles.add(setterStoreFile);
    }

    NestedSetBuilder<Artifact> classInfoFiles = NestedSetBuilder.stableOrder();
    if (classInfoFile != null) {
      classInfoFiles.add(classInfoFile);
    }

    NestedSetBuilder<Artifact> brFiles = NestedSetBuilder.stableOrder();
    if (brFile != null) {
      brFiles.add(brFile);
    }

    NestedSetBuilder<LabelJavaPackagePair> transitiveLabelAndJavaPackages =
        NestedSetBuilder.stableOrder();
    ImmutableList.Builder<LabelJavaPackagePair> labelAndJavaPackages = ImmutableList.builder();

    if (label != null && javaPackage != null) {
      LabelJavaPackagePair labelAndJavaPackage = new LabelJavaPackagePair(label, javaPackage);
      labelAndJavaPackages.add(labelAndJavaPackage);
      transitiveLabelAndJavaPackages.add(labelAndJavaPackage);
    }

    if (databindingV2ProvidersInDeps != null) {

      for (DataBindingV2ProviderApi<Artifact> p : databindingV2ProvidersInDeps) {
        DataBindingV2Provider provider = (DataBindingV2Provider) p;
        brFiles.addTransitive(provider.getTransitiveBRFiles());
        transitiveLabelAndJavaPackages.addTransitive(provider.getTransitiveLabelAndJavaPackages());
      }
    }

    if (databindingV2ProvidersInExports != null) {

      // Add all of the information from providers from exported targets, so that targets which
      // depend on this target appear to depend on the exported targets.
      for (DataBindingV2ProviderApi<Artifact> p : databindingV2ProvidersInExports) {
        DataBindingV2Provider provider = (DataBindingV2Provider) p;
        setterStoreFiles.addTransitive(provider.getSetterStores());
        classInfoFiles.addTransitive(provider.getClassInfos());
        brFiles.addTransitive(provider.getTransitiveBRFiles());
        labelAndJavaPackages.addAll(provider.getLabelAndJavaPackages());
        transitiveLabelAndJavaPackages.addTransitive(provider.getTransitiveLabelAndJavaPackages());
      }
    }

    return new DataBindingV2Provider(
        classInfoFiles.build(),
        setterStoreFiles.build(),
        brFiles.build(),
        labelAndJavaPackages.build(),
        transitiveLabelAndJavaPackages.build());
  }

  /** The provider can construct the DataBindingV2Provider provider. */
  public static class Provider extends BuiltinProvider<DataBindingV2Provider>
      implements DataBindingV2ProviderApi.Provider<Artifact> {

    private Provider() {
      super(NAME, DataBindingV2Provider.class);
    }

    @Override
    public DataBindingV2ProviderApi<Artifact> createInfo(
        Object setterStoreFile,
        Object classInfoFile,
        Object brFile,
        Object label,
        Object javaPackage,
        Sequence<?> databindingV2ProvidersInDeps, // <DataBindingV2Provider>
        Sequence<?> databindingV2ProvidersInExports) // <DataBindingV2Provider>
        throws EvalException {

      return createProvider(
          fromNoneable(setterStoreFile, Artifact.class),
          fromNoneable(classInfoFile, Artifact.class),
          fromNoneable(brFile, Artifact.class),
          fromNoneable(label, String.class),
          fromNoneable(javaPackage, String.class),
          databindingV2ProvidersInDeps == null
              ? null
              : ImmutableList.copyOf(
                  Sequence.cast(
                      databindingV2ProvidersInDeps,
                      DataBindingV2Provider.class,
                      "databinding_v2_providers_in_deps")),
          databindingV2ProvidersInExports == null
              ? null
              : ImmutableList.copyOf(
                  Sequence.cast(
                      databindingV2ProvidersInExports,
                      DataBindingV2Provider.class,
                      "databinding_v2_providers_in_exports")));
    }
  }
}
