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
// limitations under the License
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.analysis.config.CompilationMode.OPT;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDataContextApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Wraps common tools and settings used for working with Android assets, resources, and manifests.
 *
 * <p>Do not create implementation classes directly - instead, get the appropriate one from {@link
 * com.google.devtools.build.lib.rules.android.AndroidSemantics}.
 *
 * <p>The {@link Label}, {@link ActionConstructionContext}, and BusyBox {@link FilesToRunProvider}
 * are needed to create virtually all actions for working with Android data, so it makes sense to
 * bundle them together. Additionally, this class includes some common tools (such as an SDK) that
 * are used in BusyBox actions.
 */
public class AndroidDataContext implements AndroidDataContextApi {

  // Feature which would cause AndroidCompiledResourceMerger actions to pass a flag with the same
  // name to ResourceProcessorBusyBox.
  private static final String ANNOTATE_R_FIELDS_FROM_TRANSITIVE_DEPS =
      "annotate_r_fields_from_transitive_deps";

  // If specified, omit resources from transitive dependencies when generating Android R classes.
  private static final String OMIT_TRANSITIVE_RESOURCES_FROM_ANDROID_R_CLASSES =
      "android_resources_strict_deps";

  private final RuleContext ruleContext;
  private final FilesToRunProvider busybox;
  private final AndroidSdkProvider sdk;
  private final boolean persistentBusyboxToolsEnabled;
  private final boolean persistentMultiplexBusyboxToolsEnabled;
  private final boolean optOutOfResourcePathShortening;
  private final boolean optOutOfResourceNameObfuscation;
  private final boolean throwOnShrinkResources;
  private final boolean throwOnProguardApplyDictionary;
  private final boolean throwOnProguardApplyMapping;
  private final boolean throwOnResourceConflict;
  private final ImmutableMap<String, String> executionInfo;

  public static AndroidDataContext forNative(RuleContext ruleContext) throws RuleErrorException {
    return makeContext(ruleContext);
  }

  public static AndroidDataContext makeContext(RuleContext ruleContext) throws RuleErrorException {
    AndroidConfiguration androidConfig =
        ruleContext.getConfiguration().getFragment(AndroidConfiguration.class);

    ImmutableMap<String, String> executionInfo =
        TargetUtils.getExecutionInfo(ruleContext.getRule(), ruleContext.isAllowTagsPropagation());

    return new AndroidDataContext(
        ruleContext,
        ruleContext.getExecutablePrerequisite("$android_resources_busybox"),
        androidConfig.persistentBusyboxTools(),
        androidConfig.persistentMultiplexBusyboxTools(),
        AndroidSdkProvider.fromRuleContext(ruleContext),
        hasExemption(ruleContext, "allow_raw_access_to_resource_paths", false),
        hasExemption(ruleContext, "allow_resource_name_obfuscation_opt_out", false),
        !hasExemption(ruleContext, "allow_shrink_resources_attribute", true),
        !hasExemption(ruleContext, "allow_proguard_apply_dictionary", true),
        !hasExemption(ruleContext, "allow_proguard_apply_mapping", true),
        !hasExemption(ruleContext, "allow_resource_conflicts", true),
        executionInfo);
  }

  private static boolean hasExemption(
      RuleContext ruleContext, String exemptionName, boolean valueIfNoAllowlist) {
    return Allowlist.hasAllowlist(ruleContext, exemptionName)
        ? Allowlist.isAvailable(ruleContext, exemptionName)
        : valueIfNoAllowlist;
  }

  protected AndroidDataContext(
      RuleContext ruleContext,
      FilesToRunProvider busybox,
      boolean persistentBusyboxToolsEnabled,
      boolean persistentMultiplexBusyboxToolsEnabled,
      AndroidSdkProvider sdk,
      boolean optOutOfResourcePathShortening,
      boolean optOutOfResourceNameObfuscation,
      boolean throwOnShrinkResources,
      boolean throwOnProguardApplyDictionary,
      boolean throwOnProguardApplyMapping,
      boolean throwOnResourceConflict,
      ImmutableMap<String, String> executionInfo) {
    this.persistentBusyboxToolsEnabled = persistentBusyboxToolsEnabled;
    this.persistentMultiplexBusyboxToolsEnabled = persistentMultiplexBusyboxToolsEnabled;
    this.ruleContext = ruleContext;
    this.busybox = busybox;
    this.sdk = sdk;
    this.optOutOfResourcePathShortening = optOutOfResourcePathShortening;
    this.optOutOfResourceNameObfuscation = optOutOfResourceNameObfuscation;
    this.throwOnShrinkResources = throwOnShrinkResources;
    this.throwOnProguardApplyDictionary = throwOnProguardApplyDictionary;
    this.throwOnProguardApplyMapping = throwOnProguardApplyMapping;
    this.throwOnResourceConflict = throwOnResourceConflict;
    this.executionInfo = executionInfo;
  }

  public Label getLabel() {
    return ruleContext.getLabel();
  }

  public ActionConstructionContext getActionConstructionContext() {
    return ruleContext;
  }

  public RuleErrorConsumer getRuleErrorConsumer() {
    return ruleContext;
  }

  public FilesToRunProvider getBusybox() {
    return busybox;
  }

  public AndroidSdkProvider getSdk() {
    return sdk;
  }

  public ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  /*
   * Convenience methods. These are just slightly cleaner ways of doing common tasks.
   */

  /** Builds and registers a {@link SpawnAction.Builder}. */
  public void registerAction(SpawnAction.Builder spawnActionBuilder) {
    registerAction(spawnActionBuilder.build(ruleContext));
  }

  /** Registers an action. */
  public void registerAction(ActionAnalysisMetadata action) {
    ruleContext.registerAction(action);
  }

  public Artifact createOutputArtifact(SafeImplicitOutputsFunction function)
      throws InterruptedException {
    return ruleContext.getImplicitOutputArtifact(function);
  }

  public Artifact getUniqueDirectoryArtifact(String uniqueDirectorySuffix, String relative) {
    return ruleContext.getUniqueDirectoryArtifact(uniqueDirectorySuffix, relative);
  }

  public Artifact getUniqueDirectoryArtifact(String uniqueDirectorySuffix, PathFragment relative) {
    return ruleContext.getUniqueDirectoryArtifact(uniqueDirectorySuffix, relative);
  }

  public PathFragment getUniqueDirectory(PathFragment fragment) {
    return ruleContext.getUniqueDirectory(fragment);
  }

  public ArtifactRoot getBinOrGenfilesDirectory() {
    return ruleContext.getBinOrGenfilesDirectory();
  }

  public PathFragment getPackageDirectory() {
    return ruleContext.getPackageDirectory();
  }

  public AndroidConfiguration getAndroidConfig() {
    return ruleContext.getConfiguration().getFragment(AndroidConfiguration.class);
  }

  @Nullable
  public BazelAndroidConfiguration getBazelAndroidConfig() {
    return ruleContext.getConfiguration().getFragment(BazelAndroidConfiguration.class);
  }

  /** Indicates whether Busybox actions should be passed the "--debug" flag */
  public boolean useDebug() {
    return getActionConstructionContext().getConfiguration().getCompilationMode() != OPT;
  }

  public boolean isPersistentBusyboxToolsEnabled() {
    return persistentBusyboxToolsEnabled;
  }

  public boolean isPersistentMultiplexBusyboxToolsEnabled() {
    return persistentMultiplexBusyboxToolsEnabled;
  }

  public boolean optOutOfResourcePathShortening() {
    return optOutOfResourcePathShortening;
  }

  public boolean optOutOfResourceNameObfuscation() {
    return optOutOfResourceNameObfuscation;
  }

  public boolean throwOnShrinkResources() {
    return throwOnShrinkResources;
  }

  public boolean throwOnProguardApplyDictionary() {
    return throwOnProguardApplyDictionary;
  }

  public boolean throwOnProguardApplyMapping() {
    return throwOnProguardApplyMapping;
  }

  public boolean throwOnResourceConflict() {
    return throwOnResourceConflict;
  }

  public boolean annotateRFieldsFromTransitiveDeps() {
    return ruleContext.getFeatures().contains(ANNOTATE_R_FIELDS_FROM_TRANSITIVE_DEPS);
  }

  boolean omitTransitiveResourcesFromAndroidRClasses() {
    return ruleContext.getFeatures().contains(OMIT_TRANSITIVE_RESOURCES_FROM_ANDROID_R_CLASSES);
  }
}
