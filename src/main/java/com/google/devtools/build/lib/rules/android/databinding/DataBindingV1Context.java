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
import static com.google.devtools.build.lib.rules.android.databinding.DataBinding.getDataBindingExecPath;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

final class DataBindingV1Context implements DataBindingContext {

  private final ActionConstructionContext actionConstructionContext;

  DataBindingV1Context(ActionConstructionContext actionConstructionContext) {
    this.actionConstructionContext = actionConstructionContext;
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {
    consumer.accept(layoutInfoFile());
  }

  Artifact layoutInfoFile() {
    return actionConstructionContext.getUniqueDirectoryArtifact("databinding", "layout-info.zip");
  }

  @Override
  public void supplyJavaCoptsUsing(
      RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer) {
    ImmutableList.Builder<String> flags = ImmutableList.builder();
    String metadataOutputDir = getDataBindingExecPath(ruleContext).getPathString();

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
    flags.add(createProcessorFlag("xmlOutDir", getDataBindingExecPath(ruleContext).toString()));

    // Unused.
    flags.add(createProcessorFlag("exportClassListTo", "/tmp/exported_classes"));

    // The Java package for the current rule.
    flags.add(createProcessorFlag("modulePackage", AndroidCommon.getJavaPackage(ruleContext)));

    // The minimum Android SDK compatible with this rule.
    flags.add(createProcessorFlag("minApi", "14")); // TODO(gregce): update this

    // If enabled, produces cleaner output for Android Studio.
    flags.add(createProcessorFlag("printEncodedErrors", "0"));

    consumer.accept(flags.build());
  }

  @Override
  public void supplyAnnotationProcessor(
      RuleContext ruleContext, BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) {
    consumer.accept(
        JavaInfo.getProvider(
            JavaPluginInfoProvider.class,
            ruleContext.getPrerequisite(
                DataBinding.DATABINDING_ANNOTATION_PROCESSOR_ATTR, RuleConfiguredTarget.Mode.HOST)),
        DataBinding.getMetadataOutputs(ruleContext));
  }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext) {
    ImmutableList.Builder<Artifact> dataBindingJavaInputs = ImmutableList.builder();
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      dataBindingJavaInputs.add(layoutInfoFile());
    }
    for (Artifact dataBindingDepMetadata : DataBinding.getTransitiveMetadata(ruleContext, "deps")) {
      dataBindingJavaInputs.add(
          DataBinding.symlinkDepsMetadataIntoOutputTree(ruleContext, dataBindingDepMetadata));
    }
    return dataBindingJavaInputs.build();
  }

  @Override
  public ImmutableList<Artifact> addAnnotationFileToSrcs(
      ImmutableList<Artifact> srcs, RuleContext ruleContext) {
    // Add this rule's annotation processor input. If the rule doesn't have direct resources,
    // there's no direct data binding info, so there's strictly no need for annotation processing.
    // But it's still important to process the deps' .bin files so any Java class references get
    // re-referenced so they don't get filtered out of the compilation classpath by JavaBuilder
    // (which filters out classpath .jars that "aren't used": see --reduce_classpath). If data
    // binding didn't reprocess a library's data binding expressions redundantly up the dependency
    // chain (meaning each depender processes them again as if they were its own), this problem
    // wouldn't happen.
    try {
      String contents =
          ResourceFileLoader.loadResource(
              DataBinding.class, "databinding_annotation_template.txt");
      Artifact annotationFile = DataBinding
          .getDataBindingArtifact(ruleContext, "DataBindingInfo.java");
      ruleContext.registerAction(
          FileWriteAction.create(ruleContext, annotationFile, contents, false));
      return ImmutableList.<Artifact>builder().addAll(srcs).add(annotationFile).build();
    } catch (IOException e) {
      ruleContext.ruleError("Cannot load annotation processor template: " + e.getMessage());
      return ImmutableList.of();
    }
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {
    List<Artifact> dataBindingMetadataOutputs =
        Lists.newArrayList(DataBinding.getMetadataOutputs(ruleContext));
    DataBinding.maybeAddProvider(dataBindingMetadataOutputs, builder, ruleContext);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    DataBindingV1Context that = (DataBindingV1Context) o;
    return Objects.equals(actionConstructionContext, that.actionConstructionContext);
  }

  @Override
  public int hashCode() {
    return actionConstructionContext.hashCode();
  }

  @Override
  public AndroidResources processResources(AndroidResources resources) {
    return resources;
  }
}
