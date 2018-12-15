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
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidDataBindingProcessorBuilder;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

class DataBindingV2Context implements DataBindingContext {

  private final ActionConstructionContext actionContext;
  private final boolean useUpdatedArgs;
  /**
   * Annotation processing creates the following metadata files that describe how data binding is
   * applied. The full file paths include prefixes as implemented in {@link #getMetadataOutputs}.
   */
  private final List<String> metadataOutputSuffixes;

  private final String setterStoreName;

  DataBindingV2Context(ActionConstructionContext actionContext, boolean useUpdatedArgs) {
    this.actionContext = actionContext;
    this.useUpdatedArgs = useUpdatedArgs;
    this.setterStoreName = useUpdatedArgs ? "setter_store.json" : "setter_store.bin";
    metadataOutputSuffixes = ImmutableList.of(setterStoreName, "br.bin");
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {
    // In v2, The layout info file is generated in processResources below.
  }

  @Override
  public void supplyJavaCoptsUsing(RuleContext ruleContext, boolean isBinary,
      Consumer<Iterable<String>> consumer) {

    DataBindingProcessorArgsBuilder args = new DataBindingProcessorArgsBuilder(useUpdatedArgs);
    String metadataOutputDir = DataBinding.getDataBindingExecPath(ruleContext).getPathString();

    args.metadataOutputDir(metadataOutputDir);
    args.sdkDir("/not/used");
    args.binary(isBinary);
    // Unused.
    args.exportClassListTo("/tmp/exported_classes");
    args.modulePackage(AndroidCommon.getJavaPackage(ruleContext));

    // The minimum Android SDK compatible with this rule.
    // TODO(bazel-team): This probably should be based on the actual min-sdk from the manifest,
    // or an appropriate rule attribute.
    args.minApi("14");
    args.enableV2();

    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      args.classLogDir(getClassInfoFile(ruleContext));
      args.layoutInfoDir(DataBinding.getLayoutInfoFile(ruleContext));
    } else {
      // send dummy files
      args.classLogDir("/tmp/no_resources");
      args.layoutInfoDir("/tmp/no_resources");
    }
    consumer.accept(args.build());
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
        DataBinding.getMetadataOutputs(ruleContext, useUpdatedArgs, metadataOutputSuffixes);

    consumer.accept(javaPluginInfoProvider, annotationProcessorOutputs);
  }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext) {

    ImmutableList.Builder<Artifact> dataBindingJavaInputs = ImmutableList.builder();
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      dataBindingJavaInputs.add(DataBinding.getLayoutInfoFile(ruleContext));
      dataBindingJavaInputs.add(getClassInfoFile(ruleContext));
    }

    for (Artifact transitiveBRFile : getTransitiveBRFiles(ruleContext)) {
      dataBindingJavaInputs.add(
          DataBinding.symlinkDepsMetadataIntoOutputTree(ruleContext, transitiveBRFile));
    }

    for (Artifact directSetterStoreFile : getDirectSetterStoreFiles(ruleContext)) {
      dataBindingJavaInputs.add(
          DataBinding.symlinkDepsMetadataIntoOutputTree(ruleContext, directSetterStoreFile));
    }

    for (Artifact classInfo : getDirectClassInfo(ruleContext)) {
      dataBindingJavaInputs.add(
          DataBinding.symlinkDepsMetadataIntoOutputTree(ruleContext, classInfo));
    }

    return dataBindingJavaInputs.build();
  }

  private static ImmutableList<Artifact> getTransitiveBRFiles(RuleContext context) {
    ImmutableList.Builder<Artifact> brFiles = ImmutableList.builder();
    if (context.attributes().has("deps", BuildType.LABEL_LIST)) {

      Iterable<DataBindingV2Provider> providers = context.getPrerequisites(
          "deps", RuleConfiguredTarget.Mode.TARGET, DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        brFiles.addAll(provider.getTransitiveBRFiles());
      }
    }
    return brFiles.build();
  }

  private static List<Artifact> getDirectSetterStoreFiles(RuleContext context) {
    ImmutableList.Builder<Artifact> setterStoreFiles = ImmutableList.builder();
    if (context.attributes().has("deps", BuildType.LABEL_LIST)) {

      Iterable<DataBindingV2Provider> providers = context.getPrerequisites(
          "deps", RuleConfiguredTarget.Mode.TARGET, DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        setterStoreFiles.addAll(provider.getSetterStores());
      }
    }
    return setterStoreFiles.build();
  }
  
  @Override
  public ImmutableList<Artifact> getAnnotationSourceFiles(RuleContext ruleContext) {
    ImmutableList.Builder<Artifact> srcs = ImmutableList.builder();

    srcs.addAll(DataBinding.getAnnotationFile(ruleContext));
    srcs.addAll(createBaseClasses(ruleContext));

    return srcs.build();
  }

  private ImmutableList<Artifact> createBaseClasses(RuleContext ruleContext) {

    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      return ImmutableList.of(); // no resource, no base classes or class info
    }

    Artifact layoutInfo = DataBinding.getLayoutInfoFile(ruleContext);
    Artifact classInfoFile = getClassInfoFile(ruleContext);
    Artifact srcOutFile = DataBinding.getDataBindingArtifact(ruleContext, "baseClassSrc.srcjar");

    FilesToRunProvider exec = ruleContext
        .getExecutablePrerequisite(DataBinding.DATABINDING_EXEC_PROCESSOR_ATTR, Mode.HOST);

    CustomCommandLine.Builder commandLineBuilder = CustomCommandLine.builder()
        .add("GEN_BASE_CLASSES")
        .addExecPath("-layoutInfoFiles", layoutInfo)
        .add("-package", AndroidCommon.getJavaPackage(ruleContext))
        .addExecPath("-classInfoOut", classInfoFile)
        .addExecPath("-sourceOut", srcOutFile)
        .add("-zipSourceOutput", "true")
        .add("-useAndroidX", "false");

    List<Artifact> dependencyClientInfos = getDirectClassInfo(ruleContext);
    for (Artifact artifact : dependencyClientInfos) {
      commandLineBuilder.addExecPath("-dependencyClassInfoList", artifact);
    }

    Action[] action = new SpawnAction.Builder()
        .setExecutable(exec)
        .setMnemonic("GenerateDataBindingBaseClasses")
        .addInput(layoutInfo)
        .addInputs(dependencyClientInfos)
        .addOutput(classInfoFile)
        .addOutput(srcOutFile)
        .addCommandLine(commandLineBuilder.build())
        .build(ruleContext);
    ruleContext.registerAction(action);

    return ImmutableList.of(srcOutFile);
  }

  private static List<Artifact> getDirectClassInfo(RuleContext context) {
    ImmutableList.Builder<Artifact> clientInfoFiles = ImmutableList.builder();
    if (context.attributes().has("deps", BuildType.LABEL_LIST)) {

      Iterable<DataBindingV2Provider> providers = context.getPrerequisites(
          "deps", RuleConfiguredTarget.Mode.TARGET, DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        clientInfoFiles.addAll(provider.getClassInfos());
      }
    }
    return clientInfoFiles.build();
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {

    Artifact setterStore =
        DataBinding.getMetadataOutput(ruleContext, useUpdatedArgs, setterStoreName);
    Artifact br = DataBinding.getMetadataOutput(ruleContext, useUpdatedArgs, "br.bin");

    ImmutableList.Builder<Artifact> setterStores = ImmutableList.builder();
    if (setterStore != null) {
      setterStores.add(setterStore);
    }

    ImmutableList.Builder<Artifact> classInfos = ImmutableList.builder();
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      Artifact classInfo = getClassInfoFile(ruleContext);
      classInfos.add(classInfo);
    }

    // android_binary doesn't have "exports"
    if (ruleContext.attributes().has("exports", BuildType.LABEL_LIST)) {
      Iterable<DataBindingV2Provider> exportsProviders =
          ruleContext.getPrerequisites(
              "exports", RuleConfiguredTarget.Mode.TARGET, DataBindingV2Provider.PROVIDER);
      for (DataBindingV2Provider provider : exportsProviders) {
        setterStores.addAll(provider.getSetterStores());
        classInfos.addAll(provider.getClassInfos());
      }
    }

    NestedSetBuilder<Artifact> brFiles = new NestedSetBuilder<>(Order.STABLE_ORDER);
    if (br != null) {
      brFiles.add(br);
    }

    Iterable<DataBindingV2Provider> depsProviders = ruleContext.getPrerequisites(
        "deps", RuleConfiguredTarget.Mode.TARGET, DataBindingV2Provider.PROVIDER);

    for (DataBindingV2Provider provider : depsProviders) {
      brFiles.addTransitive(provider.getTransitiveBRFiles());
    }

    builder.addNativeDeclaredProvider(
        new DataBindingV2Provider(
            classInfos.build(),
            setterStores.build(),
            brFiles.build()));
  }

  @Override
  public AndroidResources processResources(
      AndroidDataContext dataContext, AndroidResources resources, String appId) {

    AndroidResources databindingProcessedResources = AndroidDataBindingProcessorBuilder.create(
        dataContext,
        resources,
        appId,
        DataBinding.getLayoutInfoFile(actionContext));

    return databindingProcessedResources;

  }

  private static Artifact getClassInfoFile(ActionConstructionContext context) {
    return context.getUniqueDirectoryArtifact("databinding", "class-info.zip");
  }
}
