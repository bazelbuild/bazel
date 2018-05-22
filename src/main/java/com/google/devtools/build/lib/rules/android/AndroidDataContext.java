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

import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

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
@SkylarkModule(
    name = "AndroidDataContext",
    doc =
        "Wraps common tools and settings used for working with Android assets, resources, and"
            + " manifests")
public class AndroidDataContext {
  private final RuleContext ruleContext;

  private final Label label;
  private final ActionConstructionContext actionConstructionContext;
  private final FilesToRunProvider busybox;
  private final AndroidSdkProvider sdk;

  public static AndroidDataContext forNative(RuleContext ruleContext) {
    return new AndroidDataContext(
        ruleContext,
        ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST),
        AndroidSdkProvider.fromRuleContext(ruleContext));
  }

  protected AndroidDataContext(
      RuleContext ruleContext, FilesToRunProvider busybox, AndroidSdkProvider sdk) {
    this.ruleContext = ruleContext;
    this.label = ruleContext.getLabel();
    this.actionConstructionContext = ruleContext;
    this.busybox = busybox;
    this.sdk = sdk;
  }

  public Label getLabel() {
    return label;
  }

  public ActionConstructionContext getActionConstructionContext() {
    return actionConstructionContext;
  }

  public FilesToRunProvider getBusybox() {
    return busybox;
  }

  public AndroidSdkProvider getSdk() {
    return sdk;
  }

  /**
   * Gets the current RuleContext.
   *
   * @deprecated RuleContext is only exposed to help migrate away from it. New code should only be
   *     written using other methods from this class.
   */
  @Deprecated
  public RuleContext getRuleContext() {
    return ruleContext;
  }

  /*
   * Convenience methods. These are just slightly cleaner ways of doing common tasks.
   */

  /** Builds and registers a {@link SpawnAction.Builder}. */
  public void registerAction(SpawnAction.Builder spawnActionBuilder) {
    registerAction(spawnActionBuilder.build(actionConstructionContext));
  }

  /** Registers one or more actions. */
  public void registerAction(ActionAnalysisMetadata... actions) {
    actionConstructionContext.registerAction(actions);
  }

  public Artifact createOutputArtifact(SafeImplicitOutputsFunction function)
      throws InterruptedException {
    return actionConstructionContext.getImplicitOutputArtifact(function);
  }

  public Artifact getUniqueDirectoryArtifact(String uniqueDirectorySuffix, String relative) {
    return actionConstructionContext.getUniqueDirectoryArtifact(uniqueDirectorySuffix, relative);
  }

  public AndroidConfiguration getAndroidConfig() {
    return actionConstructionContext.getConfiguration().getFragment(AndroidConfiguration.class);
  }
}
