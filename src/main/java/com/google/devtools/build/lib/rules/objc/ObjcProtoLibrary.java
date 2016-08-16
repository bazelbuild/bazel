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


import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * Implementation for the "objc_proto_library" rule.
 */
public class ObjcProtoLibrary implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    ProtoAttributes attributes = new ProtoAttributes(ruleContext);
    attributes.validate();

    if (attributes.hasPortableProtoFilters()) {
      return createProtobufTarget(ruleContext);
    } else {
      return createProtocolBuffers2Target(ruleContext);
    }
  }

  private ConfiguredTarget createProtobufTarget(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    ProtobufSupport protoSupport =
        new ProtobufSupport(ruleContext).registerGenerationActions().addFilesToBuild(filesToBuild);

    Optional<XcodeProvider> xcodeProvider = protoSupport.getXcodeProvider();

    new XcodeSupport(ruleContext)
        .registerActions(xcodeProvider.get())
        .addFilesToBuild(filesToBuild);

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(ObjcProvider.class, protoSupport.getObjcProvider().get())
        .addProvider(XcodeProvider.class, xcodeProvider.get())
        .build();
  }

  private ConfiguredTarget createProtocolBuffers2Target(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    ProtocolBuffers2Support protoSupport =
        new ProtocolBuffers2Support(ruleContext)
            .registerGenerationActions()
            .registerCompilationActions()
            .addFilesToBuild(filesToBuild);

    XcodeProvider xcodeProvider = protoSupport.getXcodeProvider();

    new XcodeSupport(ruleContext).registerActions(xcodeProvider).addFilesToBuild(filesToBuild);

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(ObjcProvider.class, protoSupport.getObjcProvider())
        .addProvider(XcodeProvider.class, xcodeProvider)
        .build();
  }

}
