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

import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.HDRS_TYPE;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;

import java.util.List;

/**
 * Utility code for all rules that inherit from {@link ObjcRuleClasses.ObjcBaseRule}.
 */
final class ObjcBase {
  private ObjcBase() {}

  /**
   * Provides a way to access attributes that are common to all rules that inherit from
   * {@link ObjcRuleClasses.ObjcBaseRule}.
   */
  static final class Attributes {
    private final RuleContext ruleContext;

    Attributes(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
    }

    ImmutableList<Artifact> hdrs() {
      return ruleContext.prerequisiteArtifacts("hdrs", Mode.TARGET)
          .errorsForNonMatching(HDRS_TYPE)
          .list();
    }

    Iterable<PathFragment> includes() {
      return Iterables.transform(
          ruleContext.attributes().get("includes", Type.STRING_LIST),
          PathFragment.TO_PATH_FRAGMENT);
    }

    ImmutableList<Artifact> assetCatalogs() {
      return ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET);
    }

    ImmutableList<Artifact> strings() {
      return ruleContext.getPrerequisiteArtifacts("strings", Mode.TARGET);
    }

    ImmutableList<Artifact> xibs() {
      return ruleContext.getPrerequisiteArtifacts("xibs", Mode.TARGET);
    }

    ImmutableList<Artifact> storyboards() {
      return ruleContext.getPrerequisiteArtifacts("storyboards", Mode.TARGET);
    }

    /**
     * Returns the value of the sdk_frameworks attribute plus frameworks that are included
     * automatically.
     */
    ImmutableSet<SdkFramework> sdkFrameworks() {
      ImmutableSet.Builder<SdkFramework> result = new ImmutableSet.Builder<>();
      result.addAll(ObjcRuleClasses.AUTOMATIC_SDK_FRAMEWORKS);
      for (String explicit : ruleContext.attributes().get("sdk_frameworks", Type.STRING_LIST)) {
        result.add(new SdkFramework(explicit));
      }
      return result.build();
    }

    ImmutableSet<String> sdkDylibs() {
      return ImmutableSet.copyOf(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST));
    }

    ImmutableList<Artifact> resources() {
      return ruleContext.getPrerequisiteArtifacts("resources", Mode.TARGET);
    }

    ImmutableList<Artifact> datamodels() {
      return ruleContext.getPrerequisiteArtifacts("datamodels", Mode.TARGET);
    }

    /**
     * Returns the exec paths of all header search paths that should be added to this target and
     * dependers on this target, obtained from the {@code includes} attribute.
     */
    ImmutableList<PathFragment> headerSearchPaths() {
      ImmutableList.Builder<PathFragment> paths = new ImmutableList.Builder<>();
      PathFragment packageFragment = ruleContext.getLabel().getPackageFragment();
      List<PathFragment> rootFragments = ImmutableList.of(
          packageFragment,
          ruleContext.getConfiguration().getGenfilesFragment().getRelative(packageFragment));

      Iterable<PathFragment> relativeIncludes =
          Iterables.filter(includes(), Predicates.not(PathFragment.IS_ABSOLUTE));
      for (PathFragment include : relativeIncludes) {
        for (PathFragment rootFragment : rootFragments) {
          paths.add(rootFragment.getRelative(include).normalize());
        }
      }
      return paths.build();
    }
  }

  static void registerActions(
      RuleContext ruleContext, XcodeProvider xcodeProvider, Storyboards storyboards) {
    ObjcActionsBuilder actionsBuilder = ObjcRuleClasses.actionsBuilder(ruleContext);
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    Attributes attributes = new Attributes(ruleContext);

    ObjcRuleClasses.Tools tools = new ObjcRuleClasses.Tools(ruleContext);
    actionsBuilder.registerResourceActions(
        tools,
        new ObjcActionsBuilder.StringsFiles(
            CompiledResourceFile.fromStringsFiles(intermediateArtifacts, attributes.strings())),
        new ObjcActionsBuilder.XibFiles(
            CompiledResourceFile.fromXibFiles(intermediateArtifacts, attributes.xibs())),
        Xcdatamodels.xcdatamodels(intermediateArtifacts, attributes.datamodels()));
    actionsBuilder.registerXcodegenActions(
        tools,
        ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ),
        xcodeProvider);
    for (Artifact storyboardInput : storyboards.getInputs()) {
      actionsBuilder.registerIbtoolzipAction(
          tools, storyboardInput, intermediateArtifacts.compiledStoryboardZip(storyboardInput));
    }
  }
}
