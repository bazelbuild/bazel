// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * Implementation for the "objc_proto_library" rule.
 */
public class ObjcProtoLibrary implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(final RuleContext ruleContext) throws InterruptedException {
    ProtoSupport protoSupport = new ProtoSupport(ruleContext);

    ObjcCommon.Builder commonBuilder = new ObjcCommon.Builder(ruleContext);
    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    protoSupport
        .validate()
        .addCommonOptions(commonBuilder)
        .addXcodeProviderOptions(xcodeProviderBuilder)
        .addFilesToBuild(filesToBuild)
        .registerActions();

    if (ruleContext.hasErrors()) {
      return null;
    }

    ObjcCommon common = commonBuilder.build();

    filesToBuild.addAll(common.getCompiledArchive().asSet());

    new CompilationSupport(ruleContext).registerCompileAndArchiveActions(commonBuilder.build());

    new XcodeSupport(ruleContext)
        .addFilesToBuild(filesToBuild)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), LIBRARY_STATIC)
        .addDependencies(
            xcodeProviderBuilder, new Attribute(ObjcProtoLibraryRule.PROTO_LIB_ATTR, Mode.TARGET))
        .registerActions(xcodeProviderBuilder.build());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .build();
  }
}
