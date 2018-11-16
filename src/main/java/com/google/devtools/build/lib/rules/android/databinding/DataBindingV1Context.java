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

import static com.google.devtools.build.lib.rules.android.databinding.DataBinding.createProcessorFlag;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

final class DataBindingV1Context implements DataBindingContext {

  /**
   * Annotation processing creates the following metadata files that describe how data binding is
   * applied. The full file paths include prefixes as implemented in {@link #getMetadataOutputs}.
   */
  private static final ImmutableList<String> METADATA_OUTPUT_SUFFIXES_V1 =
      ImmutableList.of("setter_store.bin", "layoutinfo.bin", "br.bin");

  private final ActionConstructionContext actionConstructionContext;

  DataBindingV1Context(ActionConstructionContext actionConstructionContext) {
    this.actionConstructionContext = actionConstructionContext;
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {
    consumer.accept(DataBinding.getLayoutInfoFile(actionConstructionContext));
  }

  @Override
  public void supplyJavaCoptsUsing(
      RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer) {

    ImmutableList.Builder<String> flags = ImmutableList.builder();
    String metadataOutputDir = DataBinding.getDataBindingExecPath(ruleContext).getPathString();

    // Directory where the annotation processor looks for deps metadata output. The annotation
    // processor automatically appends {@link DEP_METADATA_INPUT_DIR} to this path. Individual
    // files can be anywhere under this directory, recursively.
    flags.add(createProcessorFlag("bindingBuildFolder", metadataOutputDir));

    // Directory where the annotation processor should write this rule's metadata output. The
    // annotation processor automatically appends {@link METADATA_OUTPUT_DIR} to this path.
    flags.add(createProcessorFlag("generationalFileOutDir", metadataOutputDir));

    // Path to the Android SDK installation (if available).
    flags.add(createProcessorFlag("sdkDir", "/not/used"));

    // Whether the current rule is a library or binary.
    flags.add(createProcessorFlag("artifactType", isBinary ? "APPLICATION" : "LIBRARY"));

    // The path where data binding's resource processor wrote its output (the data binding XML
    // expressions). The annotation processor reads this file to translate that XML into Java.
    flags.add(createProcessorFlag("xmlOutDir", metadataOutputDir));

    // Unused.
    flags.add(createProcessorFlag("exportClassListTo", "/tmp/exported_classes"));

    // The Java package for the current rule.
    flags.add(createProcessorFlag("modulePackage", AndroidCommon.getJavaPackage(ruleContext)));

    // The minimum Android SDK compatible with this rule.
    // TODO(bazel-team): This probably should be based on the actual min-sdk from the manifest,
    // or an appropriate rule attribute.
    flags.add(createProcessorFlag("minApi", "14"));

    // If enabled, produces cleaner output for Android Studio.
    flags.add(createProcessorFlag("printEncodedErrors", "0"));

    consumer.accept(flags.build());
  }

  @Override
  public void supplyAnnotationProcessor(
      RuleContext ruleContext,
      BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) {

    JavaPluginInfoProvider javaPluginInfoProvider = JavaInfo.getProvider(
        JavaPluginInfoProvider.class,
        ruleContext.getPrerequisite(
            DataBinding.DATABINDING_ANNOTATION_PROCESSOR_ATTR, RuleConfiguredTarget.Mode.HOST));

    ImmutableList<Artifact> annotationProcessorOutputs =
        DataBinding.getMetadataOutputs(ruleContext, METADATA_OUTPUT_SUFFIXES_V1);

    consumer.accept(javaPluginInfoProvider, annotationProcessorOutputs);
  }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext) {

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
    return DataBinding.getAnnotationFile(ruleContext);
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {

    List<Artifact> dataBindingMetadataOutputs = Lists.newArrayList(
        DataBinding.getMetadataOutputs(ruleContext, METADATA_OUTPUT_SUFFIXES_V1));

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
}
