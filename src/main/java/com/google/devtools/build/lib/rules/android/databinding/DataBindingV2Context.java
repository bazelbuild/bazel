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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidDataBindingProcessorBuilder;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.DataBindingV2ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.DataBindingV2ProviderApi.LabelJavaPackagePair;
import java.util.Collection;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

class DataBindingV2Context implements DataBindingContext {

  private final ActionConstructionContext actionContext;
  private final boolean useUpdatedArgs;
  private final boolean useAndroidX;
  /**
   * Annotation processing creates the following metadata files that describe how data binding is
   * applied. The full file paths include prefixes as implemented in {@link #getMetadataOutputs}.
   */
  private final List<String> metadataOutputSuffixes;

  private final String setterStoreName;

  private final Artifact injectedLayoutInfoZip;

  DataBindingV2Context(
      ActionConstructionContext actionContext, boolean useUpdatedArgs, boolean useAndroidX) {
    this(actionContext, useUpdatedArgs, useAndroidX, null);
  }

  DataBindingV2Context(
      ActionConstructionContext actionContext,
      boolean useUpdatedArgs,
      boolean useAndroidX,
      Artifact layoutInfoZip) {
    this.actionContext = actionContext;
    this.useUpdatedArgs = useUpdatedArgs;
    this.useAndroidX = useAndroidX;
    this.setterStoreName = useUpdatedArgs ? "setter_store.json" : "setter_store.bin";
    metadataOutputSuffixes = ImmutableList.of(setterStoreName, "br.bin");
    injectedLayoutInfoZip = layoutInfoZip;
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {
    // In v2, The layout info file is generated in processResources below.
  }

  @Override
  public void supplyJavaCoptsUsing(
      RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer) {

    DataBindingProcessorArgsBuilder args = new DataBindingProcessorArgsBuilder(useUpdatedArgs);
    String metadataOutputDir = DataBinding.getDataBindingExecPath(ruleContext).getPathString();

    args.metadataOutputDir(metadataOutputDir);
    args.sdkDir("/not/used");
    args.binary(isBinary);
    // Unused.
    args.exportClassListTo("/tmp/exported_classes");
    args.modulePackage(AndroidCommon.getJavaPackage(ruleContext));
    args.directDependencyPkgs(getJavaPackagesOfDirectDependencies(ruleContext));

    // The minimum Android SDK compatible with this rule.
    // TODO(bazel-team): This probably should be based on the actual min-sdk from the manifest,
    // or an appropriate rule attribute.
    args.minApi("14");
    args.enableV2();

    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      args.classLogDir(getClassInfoFile(ruleContext));
      args.layoutInfoDir(getLayoutInfoFile());
    } else {
      // send dummy files
      args.classLogDir("/tmp/no_resources");
      args.layoutInfoDir("/tmp/no_resources");
    }
    consumer.accept(args.build());
  }

  private static Set<String> getJavaPackagesOfDirectDependencies(RuleContext ruleContext) {

    ImmutableSet.Builder<String> javaPackagesOfDirectDependencies = ImmutableSet.builder();
    if (ruleContext.attributes().has("deps", BuildType.LABEL_LIST)) {
      Iterable<DataBindingV2Provider> providers =
          ruleContext.getPrerequisites("deps", DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        for (LabelJavaPackagePair labelJavaPackagePair : provider.getLabelAndJavaPackages()) {
          javaPackagesOfDirectDependencies.add(labelJavaPackagePair.getJavaPackage());
        }
      }
    }

    return javaPackagesOfDirectDependencies.build();
  }

  @Override
  public void supplyAnnotationProcessor(
      RuleContext ruleContext, BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) {

    JavaPluginInfoProvider javaPluginInfoProvider =
        JavaInfo.getProvider(
            JavaPluginInfoProvider.class,
            ruleContext.getPrerequisite(DataBinding.DATABINDING_ANNOTATION_PROCESSOR_ATTR));

    ImmutableList<Artifact> annotationProcessorOutputs =
        DataBinding.getMetadataOutputs(ruleContext, useUpdatedArgs, metadataOutputSuffixes);

    consumer.accept(javaPluginInfoProvider, annotationProcessorOutputs);
  }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext, boolean isBinary) {

    if (isBinary) {

      // Currently, Android Databinding generates a class with a fixed name into the Java package
      // of an android_library. This means that if an android_binary depends on two
      // android_library rules with databinding in the same java package, there will be a mysterious
      // one version violation. This is an explicit check for this case so that the error is not so
      // mysterious.

      ImmutableMultimap<String, String> javaPackagesToLabel =
          getJavaPackagesWithDatabindingToLabelMap(ruleContext);

      for (Entry<String, Collection<String>> entry : javaPackagesToLabel.asMap().entrySet()) {
        if (entry.getValue().size() > 1) {
          String javaPackage = entry.getKey().isEmpty() ? "<default package>" : entry.getKey();
          ruleContext.attributeError(
              "deps",
              String.format(
                  "This target depends on multiple android_library targets "
                      + "with databinding in the same Java package. This is not supported by "
                      + "databinding and will result in class conflicts. The conflicting "
                      + "android_library targets are:\n"
                      + "  Java package %s:\n"
                      + "    %s",
                  javaPackage, Joiner.on("\n    ").join(entry.getValue())));
        }
      }
    }

    ImmutableList.Builder<Artifact> dataBindingJavaInputs = ImmutableList.builder();
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      dataBindingJavaInputs.add(getLayoutInfoFile());
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

      Iterable<DataBindingV2Provider> providers =
          context.getPrerequisites("deps", DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        brFiles.addAll(provider.getTransitiveBRFiles().toList());
      }
    }
    return brFiles.build();
  }

  private static List<Artifact> getDirectSetterStoreFiles(RuleContext context) {
    ImmutableList.Builder<Artifact> setterStoreFiles = ImmutableList.builder();
    if (context.attributes().has("deps", BuildType.LABEL_LIST)) {

      Iterable<DataBindingV2Provider> providers =
          context.getPrerequisites("deps", DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        setterStoreFiles.addAll(provider.getSetterStores());
      }
    }
    return setterStoreFiles.build();
  }

  /**
   * Collects all the labels and Java packages of the given rule and every rule that has databinding
   * in the transitive dependencies of the given rule.
   *
   * @return A multimap of Java Package (as a string) to labels which have that Java package.
   */
  private static ImmutableMultimap<String, String> getJavaPackagesWithDatabindingToLabelMap(
      RuleContext context) {

    // Since this method iterates over the labels in deps without first constructing a NestedSet,
    // multiple android_library rules could (and almost certainly will) be reached from dependencies
    // at the top-level which would produce false positives, so use a SetMultimap to avoid this.
    ImmutableMultimap.Builder<String, String> javaPackagesToLabel = ImmutableSetMultimap.builder();

    // Add this top-level rule's label and java package, e.g. for when an android_binary with
    // databinding depends on an android_library with databinding in the same java package.
    String label = context.getRule().getLabel().toString();
    String javaPackage = AndroidCommon.getJavaPackage(context);

    javaPackagesToLabel.put(javaPackage, label);

    if (context.attributes().has("deps", BuildType.LABEL_LIST)) {

      Iterable<DataBindingV2Provider> providers =
          context.getPrerequisites("deps", DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        for (LabelJavaPackagePair labelJavaPackagePair :
            provider.getTransitiveLabelAndJavaPackages().toList()) {
          javaPackagesToLabel.put(
              labelJavaPackagePair.getJavaPackage(), labelJavaPackagePair.getLabel());
        }
      }
    }

    return javaPackagesToLabel.build();
  }

  @Override
  public ImmutableList<Artifact> getAnnotationSourceFiles(RuleContext ruleContext) {
    ImmutableList.Builder<Artifact> srcs = ImmutableList.builder();

    srcs.addAll(DataBinding.getAnnotationFile(ruleContext, useAndroidX));
    srcs.addAll(createBaseClasses(ruleContext));

    return srcs.build();
  }

  private ImmutableList<Artifact> createBaseClasses(RuleContext ruleContext) {

    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      return ImmutableList.of(); // no resource, no base classes or class info
    }

    Artifact layoutInfo = getLayoutInfoFile();
    Artifact classInfoFile = getClassInfoFile(ruleContext);
    Artifact srcOutFile = DataBinding.getDataBindingArtifact(ruleContext, "baseClassSrc.srcjar");

    FilesToRunProvider exec =
        ruleContext.getExecutablePrerequisite(DataBinding.DATABINDING_EXEC_PROCESSOR_ATTR);

    CustomCommandLine.Builder commandLineBuilder =
        CustomCommandLine.builder()
            .add("GEN_BASE_CLASSES")
            .addExecPath("-layoutInfoFiles", layoutInfo)
            .add("-package", AndroidCommon.getJavaPackage(ruleContext))
            .addExecPath("-classInfoOut", classInfoFile)
            .addExecPath("-sourceOut", srcOutFile)
            .add("-zipSourceOutput", "true")
            .add("-useAndroidX", useAndroidX ? "true" : "false");

    List<Artifact> dependencyClassInfo = getDirectClassInfo(ruleContext);
    for (Artifact artifact : dependencyClassInfo) {
      commandLineBuilder.addExecPath("-dependencyClassInfoList", artifact);
    }

    Action[] action =
        new SpawnAction.Builder()
            .setExecutable(exec)
            .setMnemonic("GenerateDataBindingBaseClasses")
            .addInput(layoutInfo)
            .addInputs(dependencyClassInfo)
            .addOutput(classInfoFile)
            .addOutput(srcOutFile)
            .addCommandLine(commandLineBuilder.build())
            .build(ruleContext);
    ruleContext.registerAction(action);

    return ImmutableList.of(srcOutFile);
  }

  private static List<Artifact> getDirectClassInfo(RuleContext context) {
    ImmutableList.Builder<Artifact> classInfoFiles = ImmutableList.builder();
    if (context.attributes().has("deps", BuildType.LABEL_LIST)) {

      Iterable<DataBindingV2Provider> providers =
          context.getPrerequisites("deps", DataBindingV2Provider.PROVIDER);

      for (DataBindingV2Provider provider : providers) {
        classInfoFiles.addAll(provider.getClassInfos());
      }
    }
    return classInfoFiles.build();
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {

    Artifact setterStoreFile =
        DataBinding.getMetadataOutput(ruleContext, useUpdatedArgs, setterStoreName);

    Artifact classInfoFile;
    if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      classInfoFile = getClassInfoFile(ruleContext);
    } else {
      classInfoFile = null;
    }

    Artifact brFile = DataBinding.getMetadataOutput(ruleContext, useUpdatedArgs, "br.bin");

    String label = ruleContext.getRule().getLabel().toString();
    String javaPackage = AndroidCommon.getJavaPackage(ruleContext);

    Iterable<? extends DataBindingV2ProviderApi<Artifact>> providersFromDeps =
        ruleContext.getPrerequisites("deps", DataBindingV2Provider.PROVIDER);

    Iterable<? extends DataBindingV2ProviderApi<Artifact>> providersFromExports;
    if (ruleContext.attributes().has("exports", BuildType.LABEL_LIST)) {
      providersFromExports =
          ruleContext.getPrerequisites("exports", DataBindingV2Provider.PROVIDER);
    } else {
      providersFromExports = null;
    }

    DataBindingV2Provider provider =
        DataBindingV2Provider.createProvider(
            setterStoreFile,
            classInfoFile,
            brFile,
            label,
            javaPackage,
            providersFromDeps,
            providersFromExports);

    builder.addNativeDeclaredProvider(provider);
  }

  @Override
  public AndroidResources processResources(
      AndroidDataContext dataContext, AndroidResources resources, String appId) {

    checkArgument(injectedLayoutInfoZip == null);
    AndroidResources databindingProcessedResources =
        AndroidDataBindingProcessorBuilder.create(
            dataContext,
            resources,
            appId,
            DataBinding.getLayoutInfoFile(actionContext),
            useAndroidX);

    return databindingProcessedResources;
  }

  @Override
  public boolean usesAndroidX() {
    return useAndroidX;
  }

  private static Artifact getClassInfoFile(ActionConstructionContext context) {
    return context.getUniqueDirectoryArtifact("databinding", "class-info.zip");
  }

  private Artifact getLayoutInfoFile() {
    if (injectedLayoutInfoZip == null) {
      return DataBinding.getLayoutInfoFile(actionContext);
    }
    return injectedLayoutInfoZip;
  }
}
