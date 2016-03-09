// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarProvider.Aar;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaNeverlinkInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.ProguardLibrary;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * An implementation for the "android_library" rule.
 */
public abstract class AndroidLibrary implements RuleConfiguredTargetFactory {

  protected abstract JavaSemantics createJavaSemantics();
  protected abstract AndroidSemantics createAndroidSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    if (!AndroidSdkProvider.verifyPresence(ruleContext)) {
      return null;
    }
    checkResourceInlining(ruleContext);
    NestedSetBuilder<Aar> transitiveAars = collectTransitiveAars(ruleContext);
    NestedSet<LinkerInput> transitiveNativeLibraries =
        AndroidCommon.collectTransitiveNativeLibraries(
            AndroidCommon.collectTransitiveInfo(ruleContext, Mode.TARGET));
    NestedSet<Artifact> transitiveProguardConfigs =
        new ProguardLibrary(ruleContext).collectProguardSpecs();
    JavaCommon javaCommon = new JavaCommon(ruleContext, javaSemantics);
    AndroidCommon androidCommon = new AndroidCommon(javaCommon);

    boolean definesLocalResources =
      LocalResourceContainer.definesAndroidResources(ruleContext.attributes());
    if (definesLocalResources && !LocalResourceContainer.validateRuleContext(ruleContext)) {
      return null;
    }

    final ResourceApk resourceApk;
    if (definesLocalResources) {
      ApplicationManifest applicationManifest = androidSemantics.getManifestForRule(ruleContext);
      if (applicationManifest == null) {
        return null;
      }
      resourceApk = applicationManifest.packWithDataAndResources(
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
          ruleContext,
          true, /* isLibrary */
          ResourceDependencies.fromRuleDeps(ruleContext, JavaCommon.isNeverLink(ruleContext)),
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_SYMBOLS_TXT),
          ImmutableList.<String>of(), /* configurationFilters */
          ImmutableList.<String>of(), /* uncompressedExtensions */
          ImmutableList.<String>of(), /* densities */
          null /* applicationId */,
          null /* versionCode */,
          null /* versionName */,
          false,
          null /* proguardCfgOut */,
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_MANIFEST),
          null /* mergedResourcesOut */);
      if (ruleContext.hasErrors()) {
        return null;
      }
    } else {
      resourceApk = ResourceApk.fromTransitiveResources(
          ResourceDependencies.fromRuleResourceAndDeps(ruleContext, false /* neverlink */));
    }

    JavaTargetAttributes javaTargetAttributes = androidCommon.init(
        javaSemantics,
        androidSemantics,
        resourceApk,
        false /* addCoverageSupport */,
        true /* collectJavaCompilationArgs */);
    if (javaTargetAttributes == null) {
      return null;
    }

    Artifact classesJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR);
    Artifact aarOut = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.ANDROID_LIBRARY_AAR);

    final ResourceContainer primaryResources;
    final Aar aar;
    if (definesLocalResources) {
      primaryResources = resourceApk.getPrimaryResource();
      // applicationManifest has already been checked for nullness above in this method
      ApplicationManifest applicationManifest = androidSemantics.getManifestForRule(ruleContext);
      aar = new Aar(aarOut, applicationManifest.getManifest());
      transitiveAars.add(aar);
    } else if (AndroidCommon.getAndroidResources(ruleContext) != null) {
      primaryResources = Iterables.getOnlyElement(
          AndroidCommon.getAndroidResources(ruleContext).getDirectAndroidResources());
      aar = new Aar(aarOut, primaryResources.getManifest());
      transitiveAars.add(aar);
    } else {
      // there are no local resources and resources attribute was not specified either
      aar = null;
      ApplicationManifest applicationManifest = ApplicationManifest.generatedManifest(ruleContext);

      Artifact apk = ruleContext.getImplicitOutputArtifact(
          AndroidRuleClasses.ANDROID_RESOURCES_APK);

      String javaPackage = AndroidCommon.getJavaPackage(ruleContext);

      ResourceContainer resourceContainer = new ResourceContainer(ruleContext.getLabel(),
          javaPackage, null /* renameManifestPackage */, false /* inlinedConstants */,
          apk, applicationManifest.getManifest(),
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR),
          ImmutableList.<Artifact>of(), ImmutableList.<Artifact>of(),
          ImmutableList.<PathFragment>of(), ImmutableList.<PathFragment>of(),
          ruleContext.attributes().get("exports_manifest", Type.BOOLEAN),
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT), null);

      primaryResources = new AndroidResourcesProcessorBuilder(ruleContext)
          .setLibrary(true)
          .setApkOut(apk)
          .setRTxtOut(resourceContainer.getRTxt())
          .setManifestOut(ruleContext.getImplicitOutputArtifact(
              AndroidRuleClasses.ANDROID_LIBRARY_MANIFEST))
          .setSourceJarOut(resourceContainer.getJavaSourceJar())
          .setJavaPackage(resourceContainer.getJavaPackage())
          .withPrimary(resourceContainer)
          .withDependencies(resourceApk.getResourceDependencies())
          .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
          .build(ruleContext);
    }

    new AarGeneratorBuilder(ruleContext)
      .withPrimary(primaryResources)
      .withManifest(primaryResources.getManifest())
      .withRtxt(primaryResources.getRTxt())
      .withClasses(classesJar)
      .setAAROut(aarOut)
      .build(ruleContext);

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    androidCommon.addTransitiveInfoProviders(builder, androidSemantics, resourceApk, null,
        ImmutableList.<Artifact>of());
    androidSemantics.addTransitiveInfoProviders(
        builder, ruleContext, javaCommon, androidCommon, null);

    return builder
      .add(AndroidNativeLibraryProvider.class,
          new AndroidNativeLibraryProvider(transitiveNativeLibraries))
      .add(JavaNeverlinkInfoProvider.class,
          new JavaNeverlinkInfoProvider(androidCommon.isNeverLink()))
      .add(JavaSourceInfoProvider.class,
           JavaSourceInfoProvider.fromJavaTargetAttributes(javaTargetAttributes, javaSemantics))
      .add(JavaSourceJarsProvider.class, androidCommon.getJavaSourceJarsProvider())
      .add(AndroidCcLinkParamsProvider.class,
          new AndroidCcLinkParamsProvider(androidCommon.getCcLinkParamsStore()))
      .add(JavaPluginInfoProvider.class, JavaCommon.getTransitivePlugins(ruleContext))
      .add(ProguardSpecProvider.class, new ProguardSpecProvider(transitiveProguardConfigs))
      .addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, transitiveProguardConfigs)
      .add(AndroidLibraryAarProvider.class, new AndroidLibraryAarProvider(
                  aar, transitiveAars.build()))
      .build();
  }

  private void checkResourceInlining(RuleContext ruleContext) {
    AndroidResourcesProvider resources = AndroidCommon.getAndroidResources(ruleContext);
    if (resources == null) {
      return;
    }

    ResourceContainer container = Iterables.getOnlyElement(
        resources.getDirectAndroidResources());

    if (container.getConstantsInlined()
        && !container.getArtifacts(ResourceType.RESOURCES).isEmpty()) {
      ruleContext.ruleError("This android library has some resources assigned, so the target '"
          + resources.getLabel() + "' should have the attribute inline_constants set to 0");
    }
  }

  private NestedSetBuilder<Aar> collectTransitiveAars(RuleContext ruleContext) {
    NestedSetBuilder<Aar> builder = NestedSetBuilder.naiveLinkOrder();
    for (AndroidLibraryAarProvider library : AndroidCommon.getTransitivePrerequisites(
        ruleContext, Mode.TARGET, AndroidLibraryAarProvider.class)) {
      builder.addTransitive(library.getTransitiveAars());
    }
    return builder;
  }
}
