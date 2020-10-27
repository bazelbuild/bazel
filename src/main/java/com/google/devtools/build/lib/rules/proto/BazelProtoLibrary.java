// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;

/** The implementation of the <code>proto_library</code> rule. */
public class BazelProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws ActionConflictException, RuleErrorException, InterruptedException {
    ProtoCommon.checkSourceFilesAreInSamePackage(ruleContext);
    ProtoInfo protoInfo =
        ProtoCommon.createProtoInfo(
            ruleContext,
            ruleContext.getFragment(ProtoConfiguration.class).generatedProtosInVirtualImports());
    if (ruleContext.hasErrors()) {
      return null;
    }

    ProtoCompileActionBuilder.writeDescriptorSet(ruleContext, protoInfo, Services.ALLOW);

    Runfiles dataRunfiles =
        ProtoCommon.createDataRunfilesProvider(protoInfo.getTransitiveProtoSources(), ruleContext)
            .addArtifact(protoInfo.getDirectDescriptorSet())
            .build();

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .setFilesToBuild(
                NestedSetBuilder.create(STABLE_ORDER, protoInfo.getDirectDescriptorSet()))
            .addProvider(RunfilesProvider.withData(Runfiles.EMPTY, dataRunfiles))
            .addNativeDeclaredProvider(protoInfo);

    return builder.build();
  }
}
