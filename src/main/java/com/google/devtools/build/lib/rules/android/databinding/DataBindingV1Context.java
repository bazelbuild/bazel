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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

final class DataBindingV1Context implements DataBindingContext {

  private final ActionConstructionContext actionConstructionContext;
  private final boolean useUpdatedArgs;
  /**
   * Annotation processing creates the following metadata files that describe how data binding is
   * applied. The full file paths include prefixes as implemented in {@link #getMetadataOutputs}.
   */
  private final ImmutableList<String> metadataOutputSuffixes;

  DataBindingV1Context(
      ActionConstructionContext actionConstructionContext, boolean useUpdatedArgs) {
    this.actionConstructionContext = actionConstructionContext;
    this.useUpdatedArgs = useUpdatedArgs;
    final String setterStoreName = useUpdatedArgs ? "setter_store.json" : "setter_store.bin";
    metadataOutputSuffixes = ImmutableList.of(setterStoreName, "layoutinfo.bin", "br.bin");
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {
    consumer.accept(DataBinding.getLayoutInfoFile(actionConstructionContext));
  }

  @Override
  public void supplyJavaCoptsUsing(
      RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer) {

    DataBindingProcessorArgsBuilder args = new DataBindingProcessorArgsBuilder(useUpdatedArgs);
    String metadataOutputDir = DataBinding.getDataBindingExecPath(ruleContext).getPathString();

    args.metadataOutputDir(metadataOutputDir);
    args.sdkDir("/not/used");
    args.binary(isBinary);
    args.layoutInfoDir(metadataOutputDir);
    // Unused
    args.exportClassListTo("/tmp/exported_classes");
    args.modulePackage(AndroidCommon.getJavaPackage(ruleContext));
    // The minimum Android SDK compatible with this rule.
    // TODO(bazel-team): This probably should be based on the actual min-sdk from the manifest,
    // or an appropriate rule attribute.
    args.minApi("14");
    consumer.accept(args.build());
  }

  @Override
  public void supplyAnnotationProcessor(
      RuleContext ruleContext, BiConsumer<JavaPluginInfo, Iterable<Artifact>> consumer)
      throws RuleErrorException {

    JavaPluginInfo javaPluginInfo =
        ruleContext
            .getPrerequisite(DataBinding.DATABINDING_ANNOTATION_PROCESSOR_ATTR)
            .get(JavaPluginInfo.PROVIDER);

    ImmutableList<Artifact> annotationProcessorOutputs =
        DataBinding.getMetadataOutputs(ruleContext, useUpdatedArgs, metadataOutputSuffixes);

    consumer.accept(javaPluginInfo, annotationProcessorOutputs);
  }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext, boolean isBinary) {

    ImmutableList.Builder<Artifact> dataBindingJavaInputs = ImmutableList.builder();
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      dataBindingJavaInputs.add(DataBinding.getLayoutInfoFile(actionConstructionContext));
    }

    for (Artifact dataBindingDepMetadata : DataBinding.getTransitiveMetadata(ruleContext, "deps")) {
      dataBindingJavaInputs.add(
          DataBinding.symlinkDepsMetadataIntoOutputTree(ruleContext, dataBindingDepMetadata));
    }

    return dataBindingJavaInputs.build();
  }

  @Override
  public ImmutableList<Artifact> getAnnotationSourceFiles(RuleContext ruleContext) {
    return DataBinding.getAnnotationFile(ruleContext, /* useAndroidX= */ false);
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {

    List<Artifact> dataBindingMetadataOutputs =
        Lists.newArrayList(
            DataBinding.getMetadataOutputs(ruleContext, useUpdatedArgs, metadataOutputSuffixes));

    // Expose the data binding provider if there are outputs.
    dataBindingMetadataOutputs.addAll(DataBinding.getTransitiveMetadata(ruleContext, "exports"));
    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      // If this rule doesn't declare direct resources, no resource processing is run so no data
      // binding outputs are produced. In that case, we need to explicitly propagate data binding
      // outputs from the deps to make sure they continue up the build graph.
      dataBindingMetadataOutputs.addAll(DataBinding.getTransitiveMetadata(ruleContext, "deps"));
    }
    if (!dataBindingMetadataOutputs.isEmpty()) {
      builder.addNativeDeclaredProvider(
          new UsesDataBindingProvider(dataBindingMetadataOutputs));
    }
  }

  @Override
  public AndroidResources processResources(
      AndroidDataContext dataContext, AndroidResources resources, String appId) {
    return resources;
  }

  @Override
  public boolean usesAndroidX() {
    // AndroidX dependencies are only supported with databinding v2.
    return false;
  }
}
