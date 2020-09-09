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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.DataBindingV2ProviderApi.LabelJavaPackagePair;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

class DisabledDataBindingV2Context implements DataBindingContext {

  @Override
  public void supplyJavaCoptsUsing(RuleContext ruleContext, boolean isBinary,
      Consumer<Iterable<String>> consumer) { }

  @Override
  public void supplyAnnotationProcessor(RuleContext ruleContext,
      BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) { }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext, boolean isBinary) {
    return ImmutableList.of();
  }

  @Override
  public ImmutableList<Artifact> getAnnotationSourceFiles(RuleContext ruleContext) {
    return ImmutableList.of();
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {

    ImmutableList.Builder<Artifact> setterStores = ImmutableList.builder();
    ImmutableList.Builder<Artifact> classInfos = ImmutableList.builder();
    NestedSetBuilder<Artifact> brFiles = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<LabelJavaPackagePair> exportedLabelJavaPackages = ImmutableList.builder();
    NestedSetBuilder<LabelJavaPackagePair> targetNameAndJavaPackages =
        NestedSetBuilder.stableOrder();

    // android_binary doesn't have "exports"
    if (ruleContext.attributes().has("exports", BuildType.LABEL_LIST)) {
      Iterable<DataBindingV2Provider> exportsProviders =
          ruleContext.getPrerequisites(
              "exports", TransitionMode.TARGET, DataBindingV2Provider.PROVIDER);
      for (DataBindingV2Provider provider : exportsProviders) {
        setterStores.addAll(provider.getSetterStores());
        classInfos.addAll(provider.getClassInfos());
        brFiles.addTransitive(provider.getTransitiveBRFiles());
        exportedLabelJavaPackages.addAll(provider.getLabelAndJavaPackages());
        targetNameAndJavaPackages.addTransitive(provider.getTransitiveLabelAndJavaPackages());
      }
    }

    Iterable<DataBindingV2Provider> depsProviders =
        ruleContext.getPrerequisites("deps", TransitionMode.TARGET, DataBindingV2Provider.PROVIDER);

    for (DataBindingV2Provider provider : depsProviders) {
      brFiles.addTransitive(provider.getTransitiveBRFiles());
      targetNameAndJavaPackages.addTransitive(provider.getTransitiveLabelAndJavaPackages());
    }

    builder.addNativeDeclaredProvider(
        new DataBindingV2Provider(
            classInfos.build(),
            setterStores.build(),
            brFiles.build(),
            exportedLabelJavaPackages.build(),
            targetNameAndJavaPackages.build()));
  }

  @Override
  public AndroidResources processResources(
      AndroidDataContext dataContext, AndroidResources resources, String appId) {
    return resources;
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {  }
}
