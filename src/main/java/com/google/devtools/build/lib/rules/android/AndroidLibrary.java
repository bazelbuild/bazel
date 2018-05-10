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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarInfo.Aar;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.ProguardLibrary;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.syntax.Type;

/** An implementation for the "android_library" rule. */
public abstract class AndroidLibrary implements RuleConfiguredTargetFactory {

  protected abstract JavaSemantics createJavaSemantics();

  protected abstract AndroidSemantics createAndroidSemantics();

  protected abstract AndroidMigrationSemantics createAndroidMigrationSemantics();

  /** Checks expected rule invariants, throws rule errors if anything is set wrong. */
  private static void validateRuleContext(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    /**
     * TODO(b/14473160): Remove when deps are no longer implicitly exported.
     *
     * <p>Warn if android_library rule contains deps without srcs or locally-used resources. Such
     * deps are implicitly exported (deprecated behavior), and will soon be disallowed entirely.
     */
    if (usesDeprecatedImplicitExport(ruleContext)) {
      String message =
          "android_library will be deprecating the use of deps to export "
              + "targets implicitly. Please use android_library.exports to explicitly specify "
              + "targets this rule exports";
      AndroidConfiguration androidConfig = ruleContext.getFragment(AndroidConfiguration.class);
      if (androidConfig.allowSrcsLessAndroidLibraryDeps()) {
        ruleContext.attributeWarning("deps", message);
      } else {
        ruleContext.attributeError("deps", message);
      }
    }
  }

  /**
   * TODO(b/14473160): Remove when deps are no longer implicitly exported.
   *
   * <p>Returns true if the rule (possibly) relies on the implicit dep exports behavior.
   *
   * <p>If this returns true, then the rule *is* exporting deps implicitly, and does not have any
   * srcs or locally-used resources consuming the deps.
   *
   * <p>Else, this rule either is not using deps or has another deps-consuming attribute (src,
   * locally-used resources)
   */
  private static boolean usesDeprecatedImplicitExport(RuleContext ruleContext)
      throws RuleErrorException {
    AttributeMap attrs = ruleContext.attributes();

    if (!attrs.isAttributeValueExplicitlySpecified("deps")
        || attrs.get("deps", BuildType.LABEL_LIST).isEmpty()) {
      return false;
    }

    String[] labelListAttrs = {"srcs", "idl_srcs", "assets", "resource_files"};
    for (String attr : labelListAttrs) {
      if (attrs.isAttributeValueExplicitlySpecified(attr)
          && !attrs.get(attr, BuildType.LABEL_LIST).isEmpty()) {
        return false;
      }
    }

    boolean hasManifest = attrs.isAttributeValueExplicitlySpecified("manifest");
    boolean hasAssetsDir = attrs.isAttributeValueExplicitlySpecified("assets_dir");
    boolean hasInlineConsts =
        attrs.isAttributeValueExplicitlySpecified("inline_constants")
            && attrs.get("inline_constants", Type.BOOLEAN);
    boolean hasExportsManifest =
        attrs.isAttributeValueExplicitlySpecified("exports_manifest")
            && attrs.get("exports_manifest", BuildType.TRISTATE) == TriState.YES;

    return !(hasManifest || hasInlineConsts || hasAssetsDir || hasExportsManifest);
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    validateRuleContext(ruleContext);
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    androidSemantics.validateAndroidLibraryRuleContext(ruleContext);
    createAndroidMigrationSemantics().validateRuleContext(ruleContext);
    AndroidSdkProvider.verifyPresence(ruleContext);

    NestedSetBuilder<Artifact> proguardConfigsbuilder = NestedSetBuilder.stableOrder();
    ProguardLibrary proguardLibrary = new ProguardLibrary(ruleContext);
    proguardConfigsbuilder.addTransitive(proguardLibrary.collectProguardSpecs());
    AndroidIdlHelper.maybeAddSupportLibProguardConfigs(ruleContext, proguardConfigsbuilder);
    NestedSet<Artifact> transitiveProguardConfigs = proguardConfigsbuilder.build();

    JavaCommon javaCommon =
        AndroidCommon.createJavaCommonWithAndroidDataBinding(ruleContext, javaSemantics, true);
    javaSemantics.checkRule(ruleContext, javaCommon);
    AndroidCommon androidCommon = new AndroidCommon(javaCommon);

    AndroidConfiguration androidConfig = AndroidCommon.getAndroidConfig(ruleContext);

    boolean definesLocalResources =
        AndroidResources.definesAndroidResources(ruleContext.attributes());
    if (definesLocalResources) {
      AndroidResources.validateRuleContext(ruleContext);
    }

    // TODO(b/69668042): Always correctly apply neverlinking for resources
    boolean isNeverLink =
        JavaCommon.isNeverLink(ruleContext)
            && (definesLocalResources || androidConfig.fixedResourceNeverlinking());
    ResourceDependencies resourceDeps = ResourceDependencies.fromRuleDeps(ruleContext, isNeverLink);
    AssetDependencies assetDeps = AssetDependencies.fromRuleDeps(ruleContext, isNeverLink);

    final ResourceApk resourceApk;
    if (definesLocalResources) {
      if (androidConfig.decoupleDataProcessing()) {
        StampedAndroidManifest manifest =
            AndroidManifest.fromAttributes(ruleContext, androidSemantics).stamp(ruleContext);

        ValidatedAndroidResources resources =
            AndroidResources.from(ruleContext, "resource_files")
                .process(ruleContext, manifest, isNeverLink);

        MergedAndroidAssets assets =
            AndroidAssets.from(ruleContext).process(ruleContext, isNeverLink);

        resourceApk = ResourceApk.of(resources, assets, null, null);
      } else {
        ApplicationManifest applicationManifest =
            androidSemantics
                .getManifestForRule(ruleContext)
                .renamePackage(ruleContext, AndroidCommon.getJavaPackage(ruleContext));
        resourceApk =
            applicationManifest.packLibraryWithDataAndResources(
                ruleContext,
                resourceDeps,
                ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
                ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_MERGED_SYMBOLS),
                ruleContext.getImplicitOutputArtifact(
                    AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
                ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP),
                DataBinding.isEnabled(ruleContext)
                    ? DataBinding.getLayoutInfoFile(ruleContext)
                    : null);
      }
      if (ruleContext.hasErrors()) {
        return null;
      }
    } else {
      // Process transitive resources so we can build artifacts needed to export an aar.
      resourceApk =
          ResourceApk.processFromTransitiveLibraryData(
              ruleContext,
              resourceDeps,
              assetDeps,
              StampedAndroidManifest.createEmpty(ruleContext, /* exported = */ false));
    }

    JavaTargetAttributes javaTargetAttributes =
        androidCommon.init(
            javaSemantics,
            androidSemantics,
            resourceApk,
            /* addCoverageSupport= */ false,
            /* collectJavaCompilationArgs= */ true,
            /* isBinary= */ false,
            /* excludedRuntimeArtifacts= */ null,
            /* generateExtensionRegistry= */ false);
    if (javaTargetAttributes == null) {
      return null;
    }

    final Aar aar =
        Aar.makeAar(
            ruleContext,
            resourceApk,
            proguardLibrary.collectLocalProguardSpecs(),
            androidCommon.getClassJar());

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    androidCommon.addTransitiveInfoProviders(
        builder,
        aar.getAar(),
        resourceApk,
        null,
        ImmutableList.<Artifact>of(),
        NativeLibs.EMPTY,
        // TODO(elenairina): Use JavaCommon.isNeverlink(ruleContext) for consistency among rules.
        androidCommon.isNeverLink(),
        /* isLibrary = */ true);

    NestedSetBuilder<Artifact> transitiveResourcesJars = collectTransitiveResourceJars(ruleContext);
    if (resourceApk.getResourceJavaClassJar() != null) {
      transitiveResourcesJars.add(resourceApk.getResourceJavaClassJar());
    }

    builder
        .addNativeDeclaredProvider(
            new AndroidNativeLibsInfo(
                AndroidCommon.collectTransitiveNativeLibs(ruleContext).build()))
        .add(
            JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(javaTargetAttributes, javaSemantics))
        .add(
            AndroidCcLinkParamsProvider.class,
            AndroidCcLinkParamsProvider.create(androidCommon.getCcLinkParamsStore()))
        .add(ProguardSpecProvider.class, new ProguardSpecProvider(transitiveProguardConfigs))
        .addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, transitiveProguardConfigs)
        .add(
            AndroidLibraryResourceClassJarProvider.class,
            AndroidLibraryResourceClassJarProvider.create(transitiveResourcesJars.build()));

    if (!JavaCommon.isNeverLink(ruleContext)) {
      builder.addNativeDeclaredProvider(aar.toProvider(ruleContext, definesLocalResources));
    }

    return builder.build();
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
