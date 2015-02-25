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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;

/**
 * Support for resource processing on Objc rules.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class ResourceSupport {
  private final RuleContext ruleContext;
  private final Attributes attributes;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Iterable<Xcdatamodel> datamodels;

  /**
   * Creates a new resource support for the given context.
   */
  ResourceSupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.attributes = new Attributes(ruleContext);
    this.intermediateArtifacts = ObjcRuleClasses.intermediateArtifacts(ruleContext);
    this.datamodels = Xcdatamodels.xcdatamodels(intermediateArtifacts, attributes.datamodels());
  }

  /**
   * Registers resource generating actions (strings, storyboards, ...).
   *
   * @param storyboards storyboards defined by this rule
   *
   * @return this resource support
   */
  ResourceSupport registerActions(Storyboards storyboards) {
    ObjcActionsBuilder actionsBuilder = ObjcRuleClasses.actionsBuilder(ruleContext);

    ObjcRuleClasses.Tools tools = new ObjcRuleClasses.Tools(ruleContext);
    actionsBuilder.registerResourceActions(
        tools,
        new ObjcActionsBuilder.StringsFiles(
            CompiledResourceFile.fromStringsFiles(intermediateArtifacts, attributes.strings())),
        new XibFiles(attributes.xibs()),
        datamodels);
    for (Artifact storyboardInput : storyboards.getInputs()) {
      actionsBuilder.registerIbtoolzipAction(
          tools, storyboardInput, intermediateArtifacts.compiledStoryboardZip(storyboardInput));
    }
    return this;
  }

  /**
   * Adds common xcode settings to the given provider builder.
   *
   * @return this resource support
   */
  ResourceSupport addXcodeSettings(XcodeProvider.Builder xcodeProviderBuilder) {
    xcodeProviderBuilder.addInputsToXcodegen(Xcdatamodel.inputsToXcodegen(datamodels));
    return this;
  }

  /**
   * Validates resource attributes on this rule.
   *
   * @return this resource support
   */
  ResourceSupport validateAttributes() {
    Iterable<String> assetCatalogErrors = ObjcCommon.notInContainerErrors(
        attributes.assetCatalogs(), ObjcCommon.ASSET_CATALOG_CONTAINER_TYPE);
    for (String error : assetCatalogErrors) {
      ruleContext.attributeError("asset_catalogs", error);
    }

    Iterable<String> dataModelErrors =
        ObjcCommon.notInContainerErrors(attributes.datamodels(), Xcdatamodels.CONTAINER_TYPES);
    for (String error : dataModelErrors) {
      ruleContext.attributeError("datamodels", error);
    }

    return this;
  }

  private static class Attributes {
    private final RuleContext ruleContext;

    Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    ImmutableList<Artifact> datamodels() {
      return ruleContext.getPrerequisiteArtifacts("datamodels", Mode.TARGET).list();
    }

    ImmutableList<Artifact> xibs() {
      return ruleContext.getPrerequisiteArtifacts("xibs", Mode.TARGET)
          .errorsForNonMatching(ObjcRuleClasses.XIB_TYPE)
          .list();
    }

    ImmutableList<Artifact> strings() {
      return ruleContext.getPrerequisiteArtifacts("strings", Mode.TARGET).list();
    }

    ImmutableList<Artifact> assetCatalogs() {
      return ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET).list();
    }
  }
}
