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
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarProvider.Aar;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaNeverlinkInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.ProguardLibrary;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;

/**
 * An implementation for the "android_library" rule.
 */
public abstract class AndroidLibrary implements RuleConfiguredTargetFactory {

  protected abstract JavaSemantics createJavaSemantics();
  protected abstract AndroidSemantics createAndroidSemantics();

  /**
   * Checks expected rule invariants, throws rule errors if anything is set wrong.
   */
  private static void validateRuleContext(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("resources")
      && DataBinding.isEnabled(ruleContext)) {
      ruleContext.throwWithRuleError("Data binding doesn't work with the \"resources\" attribute. "
          + "Use \"resource_files\" instead.");
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    validateRuleContext(ruleContext);
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    if (!AndroidSdkProvider.verifyPresence(ruleContext)) {
      return null;
    }
    checkResourceInlining(ruleContext);
    NestedSetBuilder<Aar> transitiveAars = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<Artifact> transitiveAarArtifacts = NestedSetBuilder.stableOrder();
    collectTransitiveAars(ruleContext, transitiveAars, transitiveAarArtifacts);

    NestedSetBuilder<Artifact> proguardConfigsbuilder = NestedSetBuilder.stableOrder();
    proguardConfigsbuilder.addTransitive(new ProguardLibrary(ruleContext).collectProguardSpecs());
    AndroidIdlHelper.maybeAddSupportLibProguardConfigs(ruleContext, proguardConfigsbuilder);
    NestedSet<Artifact> transitiveProguardConfigs = proguardConfigsbuilder.build();

    JavaCommon javaCommon =
        AndroidCommon.createJavaCommonWithAndroidDataBinding(ruleContext, javaSemantics, true);
    javaSemantics.checkRule(ruleContext, javaCommon);
    AndroidCommon androidCommon = new AndroidCommon(javaCommon);

    boolean definesLocalResources =
      LocalResourceContainer.definesAndroidResources(ruleContext.attributes());
    if (definesLocalResources) {
      LocalResourceContainer.validateRuleContext(ruleContext);
    }

    final ResourceApk resourceApk;
    if (definesLocalResources) {
      ApplicationManifest applicationManifest = androidSemantics.getManifestForRule(ruleContext)
          .renamePackage(ruleContext, AndroidCommon.getJavaPackage(ruleContext));
      resourceApk =
          applicationManifest.packLibraryWithDataAndResources(
              ruleContext,
              null /* resourceApk, optional */,
              ResourceDependencies.fromRuleDeps(ruleContext, JavaCommon.isNeverLink(ruleContext)),
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_MERGED_SYMBOLS),
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP),
              DataBinding.isEnabled(ruleContext)
                  ? DataBinding.getLayoutInfoFile(ruleContext)
                  : null);
      if (ruleContext.hasErrors()) {
        return null;
      }
    } else {
      resourceApk = ResourceApk.fromTransitiveResources(
          ResourceDependencies.fromRuleResourceAndDeps(ruleContext, false /* neverlink */));
    }

    AndroidConfiguration androidConfig = ruleContext.getFragment(AndroidConfiguration.class);
    if (!androidConfig.allowSrcsLessAndroidLibraryDeps()
        && !definesLocalResources
        && ruleContext.attributes().get("srcs", BuildType.LABEL_LIST).isEmpty()
        && ruleContext.attributes().get("idl_srcs", BuildType.LABEL_LIST).isEmpty()
        && !ruleContext.attributes().get("deps", BuildType.LABEL_LIST).isEmpty()) {
      ruleContext.attributeError("deps", "deps not allowed without srcs; move to exports?");
    }

    JavaTargetAttributes javaTargetAttributes = androidCommon.init(
        javaSemantics,
        androidSemantics,
        resourceApk,
        false /* addCoverageSupport */,
        true /* collectJavaCompilationArgs */,
        false /* isBinary */,
        androidConfig.includeLibraryResourceJars());
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
      aar = Aar.create(aarOut, applicationManifest.getManifest());
      addAarToProvider(aar, transitiveAars, transitiveAarArtifacts);
    } else if (AndroidCommon.getAndroidResources(ruleContext) != null) {
      primaryResources = Iterables.getOnlyElement(
          AndroidCommon.getAndroidResources(ruleContext).getDirectAndroidResources());
      aar = Aar.create(aarOut, primaryResources.getManifest());
      addAarToProvider(aar, transitiveAars, transitiveAarArtifacts);
    } else {
      // there are no local resources and resources attribute was not specified either
      aar = null;
      ApplicationManifest applicationManifest = ApplicationManifest.generatedManifest(ruleContext)
          .renamePackage(ruleContext, AndroidCommon.getJavaPackage(ruleContext));

      String javaPackage = AndroidCommon.getJavaPackage(ruleContext);

      ResourceContainer resourceContainer =
          ResourceContainer.builder()
              .setLabel(ruleContext.getLabel())
              .setJavaPackageFromString(javaPackage)
              .setManifest(applicationManifest.getManifest())
              .setJavaSourceJar(
                  ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR))
              .setManifestExported(AndroidCommon.getExportsManifest(ruleContext))
              .setRTxt(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT))
              .build();

      primaryResources = new AndroidResourcesProcessorBuilder(ruleContext)
          .setLibrary(true)
          .setRTxtOut(resourceContainer.getRTxt())
          .setManifestOut(ruleContext.getImplicitOutputArtifact(
              AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST))
          .setSourceJarOut(resourceContainer.getJavaSourceJar())
          .setJavaPackage(resourceContainer.getJavaPackage())
          .withPrimary(resourceContainer)
          .withDependencies(resourceApk.getResourceDependencies())
          .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
          .setThrowOnResourceConflict(
              ruleContext.getFragment(AndroidConfiguration.class).throwOnResourceConflict())
          .build(ruleContext);
    }

    new AarGeneratorBuilder(ruleContext)
      .withPrimary(primaryResources)
      .withManifest(aar != null ? aar.getManifest() : primaryResources.getManifest())
      .withRtxt(primaryResources.getRTxt())
      .withClasses(classesJar)
      .setAAROut(aarOut)
      .setThrowOnResourceConflict(
          ruleContext.getFragment(AndroidConfiguration.class).throwOnResourceConflict())
      .build(ruleContext);

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    androidCommon.addTransitiveInfoProviders(
        builder,
        androidSemantics,
        aarOut,
        resourceApk,
        null,
        ImmutableList.<Artifact>of(),
        NativeLibs.EMPTY);

    NestedSetBuilder<Artifact> transitiveResourcesJars = collectTransitiveResourceJars(ruleContext);
    if (androidCommon.getResourceClassJar() != null) {
      transitiveResourcesJars.add(androidCommon.getResourceClassJar());
    }

    builder
        .addProvider(
            NativeLibsZipsProvider.class,
            new NativeLibsZipsProvider(
                AndroidCommon.collectTransitiveNativeLibsZips(ruleContext).build()))
        .add(
            JavaNeverlinkInfoProvider.class,
            new JavaNeverlinkInfoProvider(androidCommon.isNeverLink()))
        .add(
            JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(javaTargetAttributes, javaSemantics))
        .add(
            AndroidCcLinkParamsProvider.class,
            AndroidCcLinkParamsProvider.create(androidCommon.getCcLinkParamsStore()))
        .add(JavaPluginInfoProvider.class, JavaCommon.getTransitivePlugins(ruleContext))
        .add(ProguardSpecProvider.class, new ProguardSpecProvider(transitiveProguardConfigs))
        .addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, transitiveProguardConfigs)
        .add(
            AndroidLibraryResourceClassJarProvider.class,
            AndroidLibraryResourceClassJarProvider.create(transitiveResourcesJars.build()));

    if (!JavaCommon.isNeverLink(ruleContext)) {
      builder.add(
          AndroidLibraryAarProvider.class,
          AndroidLibraryAarProvider.create(
              aar, transitiveAars.build(), transitiveAarArtifacts.build()));
    }

    return builder.build();
  }

  private void addAarToProvider(
      Aar aar,
      NestedSetBuilder<Aar> transitiveAars,
      NestedSetBuilder<Artifact> transitiveAarArtifacts) {
    transitiveAars.add(aar);
    if (aar.getAar() != null) {
      transitiveAarArtifacts.add(aar.getAar());
    }
    if (aar.getManifest() != null) {
      transitiveAarArtifacts.add(aar.getManifest());
    }
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

  private void collectTransitiveAars(
      RuleContext ruleContext,
      NestedSetBuilder<Aar> transitiveAars,
      NestedSetBuilder<Artifact> transitiveAarArtifacts) {
    for (AndroidLibraryAarProvider library : AndroidCommon.getTransitivePrerequisites(
        ruleContext, Mode.TARGET, AndroidLibraryAarProvider.class)) {
      transitiveAars.addTransitive(library.getTransitiveAars());
      transitiveAarArtifacts.addTransitive(library.getTransitiveAarArtifacts());
    }
  }

  private NestedSetBuilder<Artifact> collectTransitiveResourceJars(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.naiveLinkOrder();
    Iterable<AndroidLibraryResourceClassJarProvider> providers =
        AndroidCommon.getTransitivePrerequisites(
            ruleContext, Mode.TARGET, AndroidLibraryResourceClassJarProvider.class);
    for (AndroidLibraryResourceClassJarProvider resourceJarProvider : providers) {
      builder.addTransitive(resourceJarProvider.getResourceClassJars());
    }
    return builder;
  }
}

