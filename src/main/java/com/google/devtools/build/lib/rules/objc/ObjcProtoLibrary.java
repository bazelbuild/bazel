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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;

/**
 * Implementation for the "objc_proto_library" rule.
 */
public class ObjcProtoLibrary implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    new ProtoAttributes(ruleContext).validate();
    return createProtobufTarget(ruleContext);
  }

  private ConfiguredTarget createProtobufTarget(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    Iterable<ProtoInfo> protoInfos =
        ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoInfo.class);

    Iterable<ObjcProtoProvider> objcProtoProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProtoProvider.SKYLARK_CONSTRUCTOR);

    ProtobufSupport protoSupport =
        new ProtobufSupport(
                ruleContext,
                ruleContext.getConfiguration(),
                protoInfos,
                objcProtoProviders,
                getPortableProtoFilters(ruleContext, objcProtoProviders, protoInfos))
            .registerGenerationActions()
            .addFilesToBuild(filesToBuild);

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addNativeDeclaredProvider(protoSupport.getObjcProvider().get())
        .build();
  }

  private static NestedSet<Artifact> getPortableProtoFilters(
      RuleContext ruleContext,
      Iterable<ObjcProtoProvider> objcProtoProviders,
      Iterable<ProtoInfo> protoInfos) {
    ProtoAttributes attributes = new ProtoAttributes(ruleContext);
    NestedSetBuilder<Artifact> portableProtoFilters = NestedSetBuilder.stableOrder();

    portableProtoFilters.addTransitive(
        ProtobufSupport.getTransitivePortableProtoFilters(objcProtoProviders));

    // If this target specifies filters, use those. If not, generate a filter only if there are
    // direct proto_library targets, and generate a filter only for those files.
    if (attributes.hasPortableProtoFilters()) {
      portableProtoFilters.addAll(attributes.getPortableProtoFilters());
    } else if (!Iterables.isEmpty(protoInfos)) {
      Artifact generatedFilter = ProtobufSupport.getGeneratedPortableFilter(ruleContext,
          ruleContext.getConfiguration());
      ProtobufSupport.registerPortableFilterGenerationAction(
          ruleContext, generatedFilter, protoInfos);
      portableProtoFilters.add(generatedFilter);
    }

    return portableProtoFilters.build();
  }
}
