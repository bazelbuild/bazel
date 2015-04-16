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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;

/**
 * Support for Objc rule types that export an Xcode provider or generate xcode project files.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
public final class XcodeSupport {

  /**
   * Template for a target's xcode project.
   */
  public static final SafeImplicitOutputsFunction PBXPROJ =
      fromTemplates("%{name}.xcodeproj/project.pbxproj");

  private final RuleContext ruleContext;

  /**
   * Creates a new xcode support for the given context.
   */
  XcodeSupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Adds xcode project files to the given builder.
   *
   * @return this xcode support
   */
  XcodeSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild) {
    filesToBuild.add(ruleContext.getImplicitOutputArtifact(PBXPROJ));
    return this;
  }

  /**
   * Adds a dummy source file to the Xcode target. This is needed if the target does not have any
   * source files but Xcode requires one.
   *
   * @return this xcode support
   */
  XcodeSupport addDummySource(XcodeProvider.Builder xcodeProviderBuilder) {
    xcodeProviderBuilder.addAdditionalSources(
        ruleContext.getPrerequisiteArtifact("$dummy_source", Mode.TARGET));
    return this;
  }

  /**
   * Registers actions that generate the rule's Xcode project.
   *
   * @param xcodeProvider information about this rule's xcode settings and that of its dependencies
   * @return this xcode support
   */
  XcodeSupport registerActions(XcodeProvider xcodeProvider) {
    ObjcActionsBuilder actionsBuilder = ObjcRuleClasses.actionsBuilder(ruleContext);
    actionsBuilder.registerXcodegenActions(
        new ObjcRuleClasses.Tools(ruleContext),
        ruleContext.getImplicitOutputArtifact(XcodeSupport.PBXPROJ),
        XcodeProvider.Project.fromTopLevelTarget(xcodeProvider));
    return this;
  }

  /**
   * Adds common xcode settings to the given provider builder.
   *
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param productType type of this rule's Xcode target
   *
   * @return this xcode support
   */
  XcodeSupport addXcodeSettings(XcodeProvider.Builder xcodeProviderBuilder,
      ObjcProvider objcProvider, XcodeProductType productType) {
    xcodeProviderBuilder
        .setLabel(ruleContext.getLabel())
        .setArchitecture(ObjcRuleClasses.objcConfiguration(ruleContext).getIosCpu())
        .setObjcProvider(objcProvider)
        .setProductType(productType);
    return this;
  }

  /**
   * Adds dependencies to the given provider builder from the given attribute.
   *
   * @return this xcode support
   */
  XcodeSupport addDependencies(Builder xcodeProviderBuilder, Attribute attribute) {
    xcodeProviderBuilder.addPropagatedDependencies(
        ruleContext.getPrerequisites(
            attribute.getName(), attribute.getAccessMode(), XcodeProvider.class),
        ObjcRuleClasses.objcConfiguration(ruleContext));
    return this;
  }

  /**
   * Adds non-propagated dependencies to the given provider builder from the given attribute.
   *
   * <p>A non-propagated dependency will not be linked into the final app bundle and can only serve
   * as a compile-only dependency for its direct dependent.
   *
   * @return this xcode support
   */
  XcodeSupport addNonPropagatedDependencies(Builder xcodeProviderBuilder, Attribute attribute) {
    xcodeProviderBuilder.addNonPropagatedDependencies(
        ruleContext.getPrerequisites(
            attribute.getName(), attribute.getAccessMode(), XcodeProvider.class),
        ObjcRuleClasses.objcConfiguration(ruleContext));
    return this;
  }
}
