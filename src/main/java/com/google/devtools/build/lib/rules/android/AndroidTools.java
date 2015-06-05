// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.BaseJavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.Jvm;

import java.util.List;

/** A common interface for all the tools used by {@code android_*} rules. */
public class AndroidTools {
  private final RuleContext ruleContext;
  private final Artifact dxJar;
  private final FilesToRunProvider aapt;
  private final Artifact apkBuilderTool;
  private final Artifact aidlTool;
  private final Artifact frameworkAidl;
  private final FilesToRunProvider adb;
  private final FilesToRunProvider toolRunner;
  private final FilesToRunProvider aaptJavaGenerator;
  private final FilesToRunProvider apkGenerator;
  private final FilesToRunProvider resourceProcessor;
  private final FilesToRunProvider aarGenerator;
  private final FilesToRunProvider zipalign;
  private final FilesToRunProvider proguard;
  private final Artifact androidJar;
  private final Artifact shrinkedAndroidJar;
  private final Artifact annotationsJar;
  private final Artifact mainDexClasses;
  private final AndroidSdkProvider androidSdk;

  public static AndroidTools fromRuleContext(RuleContext ruleContext) {
    TransitiveInfoCollection androidSdkDep =
        ruleContext.getPrerequisite(":android_sdk", Mode.TARGET);
    AndroidSdkProvider androidSdk = androidSdkDep == null
        ? null
        : androidSdkDep.getProvider(AndroidSdkProvider.class);

    return new AndroidTools(
        ruleContext,
        androidSdk,
        getOptionalArtifact(ruleContext, "$android_dx_jar"),
        getOptionalArtifact(ruleContext, "$android_jar", Mode.TARGET),
        getOptionalArtifact(ruleContext, "$shrinked_android"),
        getOptionalArtifact(ruleContext, "$android_annotations_jar"),
        getOptionalArtifact(ruleContext, "$multidex_keep_classes"),
        getOptionalArtifact(ruleContext, "$android_apkbuilder_tool"),
        getOptionalArtifact(ruleContext, "$android_aidl_tool"),
        getOptionalArtifact(ruleContext, "$android_aidl_framework"),
        getOptionalToolFromArtifact(ruleContext, "$android_aapt"),
        getOptionalToolFromArtifact(ruleContext, "$adb"),
        getOptionalTool(ruleContext, "$android_tool_runner"),
        getOptionalTool(ruleContext, "$android_aapt_java_generator"),
        getOptionalTool(ruleContext, "$android_aapt_apk_generator"),
        getOptionalTool(ruleContext, ":android_resources_processor"),
        getOptionalTool(ruleContext, ":android_aar_generator"),
        getOptionalTool(ruleContext, "$zipalign_tool"),
        getOptionalTool(ruleContext, ":proguard"));
  }

  private static Artifact getOptionalArtifact(RuleContext ruleContext, String attribute) {
    return getOptionalArtifact(ruleContext, attribute, Mode.HOST);
  }

  private static Artifact getOptionalArtifact(
      RuleContext ruleContext, String attribute, Mode mode) {
    if (!ruleContext.getRule().isAttrDefined(attribute, Type.LABEL)) {
      return null;
    }

    List<Artifact> prerequisites =
        ruleContext.getPrerequisiteArtifacts(attribute, mode).list();

    if (prerequisites.isEmpty()) {
      return null;
    } else if (prerequisites.size() == 1) {
      return prerequisites.get(0);
    } else {
      ruleContext.attributeError(attribute, "expected a single artifact");
      return null;
    }
  }

  private static FilesToRunProvider getOptionalToolFromArtifact(
      RuleContext ruleContext, String attribute) {
    if (!ruleContext.getRule().isAttrDefined(attribute, Type.LABEL)) {
      return null;
    }

    Artifact artifact = getOptionalArtifact(ruleContext, attribute);
    if (artifact == null) {
      return null;
    }

    return FilesToRunProvider.fromSingleArtifact(
        ruleContext.attributes().get(attribute, Type.LABEL),
        artifact);
  }

  private static FilesToRunProvider getOptionalTool(RuleContext ruleContext, String attribute) {
    if (!ruleContext.getRule().isAttrDefined(attribute, Type.LABEL)) {
      return null;
    }

    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite(attribute, Mode.HOST);
    if (prerequisite == null) {
      return null;
    }

    return prerequisite.getProvider(FilesToRunProvider.class);
  }

  public AndroidTools(
      RuleContext ruleContext,
      AndroidSdkProvider androidSdk,
      Artifact dxJar,
      Artifact androidJar,
      Artifact shrinkedAndroidJar,
      Artifact annotationsJar,
      Artifact mainDexClasses,
      Artifact apkBuilderTool,
      Artifact aidlTool,
      Artifact frameworkAidl,
      FilesToRunProvider aapt,
      FilesToRunProvider adb,
      FilesToRunProvider toolRunner,
      FilesToRunProvider aaptJavaGenerator,
      FilesToRunProvider apkGenerator,
      FilesToRunProvider resourceProcessor,
      FilesToRunProvider aarGenerator,
      FilesToRunProvider zipalign,
      FilesToRunProvider proguard) {
    this.ruleContext = ruleContext;
    this.androidSdk = androidSdk;
    this.dxJar = dxJar;
    this.androidJar = androidJar;
    this.shrinkedAndroidJar = shrinkedAndroidJar;
    this.mainDexClasses = mainDexClasses;
    this.aapt = aapt;
    this.annotationsJar = annotationsJar;
    this.apkBuilderTool = apkBuilderTool;
    this.aidlTool = aidlTool;
    this.frameworkAidl = frameworkAidl;
    this.adb = adb;
    this.toolRunner = toolRunner;
    this.aaptJavaGenerator = aaptJavaGenerator;
    this.apkGenerator = apkGenerator;
    this.resourceProcessor = resourceProcessor;
    this.aarGenerator = aarGenerator;
    this.zipalign = zipalign;
    this.proguard = proguard;
  }

  public Artifact getFrameworkAidl() {
    return androidSdk != null ? androidSdk.getFrameworkAidl() : frameworkAidl;
  }

  public Artifact getAndroidJar() {
    return androidSdk != null ? androidSdk.getAndroidJar() : androidJar;
  }

  public Artifact getShrinkedAndroidJar() {
    return androidSdk != null ? androidSdk.getShrinkedAndroidJar() : shrinkedAndroidJar;
  }

  public Artifact getAnnotationsJar() {
    return androidSdk != null ? androidSdk.getAnnotationsJar() : annotationsJar;
  }

  public Artifact getMainDexClasses() {
    return androidSdk != null ? androidSdk.getMainDexClasses() : mainDexClasses;
  }

  public FilesToRunProvider getAapt() {
    return androidSdk != null ? androidSdk.getAapt() : aapt;
  }

  public FilesToRunProvider getAdb() {
    return androidSdk != null ? androidSdk.getAdb() : adb;
  }

  public FilesToRunProvider getToolRunner() {
    return toolRunner;
  }

  public FilesToRunProvider getAaptJavaGenerator() {
    return aaptJavaGenerator;
  }

  public FilesToRunProvider getApkGenerator() {
    return apkGenerator;
  }

  public FilesToRunProvider getAndroidResourceProcessor() {
    return resourceProcessor;
  }

  public FilesToRunProvider getAarGenerator() {
    return aarGenerator;
  }

  public FilesToRunProvider getZipalign() {
    return androidSdk != null ? androidSdk.getZipalign() : zipalign;
  }

  public FilesToRunProvider getProguard() {
    return androidSdk != null ? androidSdk.getProguard() : proguard;
  }

  /**
   * Creates a {@link com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder} that has
   * its executable already set to invoke dx.
   */
  public SpawnAction.Builder dxAction(AndroidSemantics semantics) {
    return androidSdk != null
        ? new SpawnAction.Builder()
            .setExecutable(androidSdk.getDx())
        : new SpawnAction.Builder()
            .addTransitiveInputs(BaseJavaCompilationHelper.getHostJavabaseInputs(ruleContext))
            .setExecutable(ruleContext.getHostConfiguration()
                .getFragment(Jvm.class).getJavaExecutable())
            .addArguments(semantics.getDxJvmArguments())
            .addArgument("-jar")
            .addInputArgument(dxJar);
  }

  /**
   * Creates a {@link com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder} that has
   * its executable already set to invoke the main dex list creator.
   */
  public Action[] mainDexListAction(Artifact jar, Artifact strippedJar, Artifact mainDexList) {
    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setMnemonic("MainDexClasses")
        .setProgressMessage("Generating main dex classes list");

    if (androidSdk != null) {
      return builder
          .setExecutable(androidSdk.getMainDexListCreator())
          .addOutputArgument(mainDexList)
          .addInputArgument(strippedJar)
          .addInputArgument(jar)
          .addArguments(ruleContext.getTokenizedStringListAttr("main_dex_list_opts"))
          .build(ruleContext);
    } else {
      StringBuilder shellCommandBuilder = new StringBuilder()
          .append(ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable()
              .getPathString())
          .append(" -cp ").append(dxJar.getExecPathString())
          .append(" ").append(AndroidBinary.MAIN_DEX_CLASS_BUILDER);
      for (String opt : ruleContext.getTokenizedStringListAttr("main_dex_list_opts")) {
        shellCommandBuilder.append(" ").append(opt);
      }
      shellCommandBuilder
          .append(" ").append(strippedJar.getExecPathString())
          .append(" ").append(jar.getExecPathString())
          .append(" >").append(mainDexList.getExecPathString());

      return builder
          .addInput(strippedJar)
          .addInput(jar)
          .addInput(dxJar)
          .addTransitiveInputs(BaseJavaCompilationHelper.getHostJavabaseInputs(ruleContext))
          .addOutput(mainDexList)
          .setShellCommand(shellCommandBuilder.toString())
          .build(ruleContext);
    }
  }

    /**
   * Creates a {@link com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder} that has
   * its executable already set to invoke apkbuilder.
   */
  public SpawnAction.Builder apkBuilderAction() {
    return androidSdk != null
        ? new SpawnAction.Builder()
            .setExecutable(androidSdk.getApkBuilder())
        : new SpawnAction.Builder()
          .setExecutable(
              ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable())
          .addTransitiveInputs(JavaCompilationHelper.getHostJavabaseInputs(ruleContext))
          .addArgument("-jar")
          .addInputArgument(apkBuilderTool);
  }

  /**
   * Creates a {@link com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder} that has
   * its executable already set to invoke aidl.
   */
  public SpawnAction.Builder aidlAction() {
    return androidSdk != null
        ? new SpawnAction.Builder()
            .setExecutable(androidSdk.getAidl())
        : new SpawnAction.Builder()
            // Note the below may be an overapproximation of the actual runfiles, due to
            // "conditional artifacts" (see Runfiles.PruningManifest).
            // TODO(bazel-team): When using getFilesToRun(), the middleman is
            // not expanded. Fix by providing code to expand and use getFilesToRun here.
            .addInputs(toolRunner.getRunfilesSupport().getRunfilesArtifactsWithoutMiddlemen())
            .setExecutable(toolRunner.getExecutable())
            .addInputArgument(aidlTool);
  }
}