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
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ProtoSupport.TargetType;

/**
 * Implementation for the "objc_proto_library" rule.
 */
public class ObjcProtoLibrary implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    ProtoSupport protoSupport =
        new ProtoSupport(ruleContext, TargetType.PROTO_TARGET)
            .validate()
            .addXcodeProviderOptions(xcodeProviderBuilder)
            .addFilesToBuild(filesToBuild)
            .registerActions();

    if (ruleContext.hasErrors()) {
      return null;
    }

    ObjcCommon common = protoSupport.getCommon();

    filesToBuild.addAll(common.getCompiledArchive().asSet());

    new XcodeSupport(ruleContext)
        .addFilesToBuild(filesToBuild)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), LIBRARY_STATIC)
        .addDependencies(
            xcodeProviderBuilder, new Attribute(ObjcRuleClasses.PROTO_LIB_ATTR, Mode.TARGET))
        .registerActions(xcodeProviderBuilder.build());

    boolean usesProtobufLibrary = protoSupport.usesProtobufLibrary();

    boolean experimentalAutoUnion =
        ObjcRuleClasses.objcConfiguration(ruleContext).experimentalAutoTopLevelUnionObjCProtos();

    // If the experimental flag is not set, or if it's set and doesn't use the protobuf library,
    // register the compilation actions, as the output needs to be linked in the final binary.
    if (!experimentalAutoUnion || !usesProtobufLibrary) {
      new CompilationSupport(ruleContext).registerCompileAndArchiveActions(common);
    }

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .build();
  }
}
