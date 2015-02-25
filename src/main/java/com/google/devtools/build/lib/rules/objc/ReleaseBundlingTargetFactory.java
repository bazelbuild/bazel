// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;

/**
 * Base class for rules that bundle releases.
 */
public abstract class ReleaseBundlingTargetFactory implements RuleConfiguredTargetFactory {

  /**
   * Indicates whether a target factory should export an {@link ObjcProvider} containing itself as
   * a nested bundle.
   */
  protected enum ExposeAsNestedBundle { YES, NO }

  private final String bundleDirFormat;
  private final XcodeProductType xcodeProductType;
  private final ExposeAsNestedBundle exposeAsNestedBundle;

  /**
   * @param bundleDirFormat format string representing the bundle's directory with a single
   *     placeholder for the target name (e.g. {@code "Payload/%s.app"})
   * @param exposeAsNestedBundle whether to export an {@link ObjcProvider} with this target as a
   *    nested bundle
   */
  public ReleaseBundlingTargetFactory(String bundleDirFormat,
      XcodeProductType xcodeProductType,
      ExposeAsNestedBundle exposeAsNestedBundle) {
    this.bundleDirFormat = bundleDirFormat;
    this.xcodeProductType = xcodeProductType;
    this.exposeAsNestedBundle = exposeAsNestedBundle;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext);

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    ReleaseBundlingSupport releaseBundlingSupport = new ReleaseBundlingSupport(
        ruleContext, common.getObjcProvider(), optionsProvider(ruleContext),
        LinkedBinary.DEPENDENCIES_ONLY, bundleDirFormat);
    releaseBundlingSupport
        .registerActions()
        .addXcodeSettings(xcodeProviderBuilder)
        .addFilesToBuild(filesToBuild)
        .validateAttributes();

    new XcodeSupport(ruleContext)
        .addFilesToBuild(filesToBuild)
        .addXcodeSettings(
            xcodeProviderBuilder, common.getObjcProvider(), xcodeProductType)
        .addDummySource(xcodeProviderBuilder)
        .addDependencies(xcodeProviderBuilder, "binary")
        .registerActions(xcodeProviderBuilder.build());


    Optional<ObjcProvider> exposedObjcProvider;
    if (exposeAsNestedBundle == ExposeAsNestedBundle.YES) {
      exposedObjcProvider = Optional.of(new ObjcProvider.Builder()
          .add(MERGE_ZIP, ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA))
          .build());
    } else {
      exposedObjcProvider = Optional.absent();
    }

    return common.configuredTarget(
        filesToBuild.build(),
        Optional.of(xcodeProviderBuilder.build()),
        exposedObjcProvider,
        Optional.<XcTestAppProvider>absent(),
        Optional.<J2ObjcSrcsProvider>absent());
  }

  /**
   * Returns a provider based on this rule's options and those of its option-providing dependencies.
   */
  protected abstract OptionsProvider optionsProvider(RuleContext ruleContext);

  private ObjcCommon common(RuleContext ruleContext) {
    return new ObjcCommon.Builder(ruleContext)
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .addDepObjcProviders(
            ruleContext.getPrerequisites("binary", Mode.TARGET, ObjcProvider.class))
        .build();
  }
}
