// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;

/** Implements {code proto_lang_toolchain}. */
public class ProtoLangToolchain implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
    for (ProtoInfo protoInfo :
        ruleContext.getPrerequisites("blacklisted_protos", ProtoInfo.PROVIDER)) {
      blacklistedProtos.addTransitive(protoInfo.getOriginalTransitiveProtoSources());
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(
            ProtoLangToolchainProvider.create(
                ruleContext.attributes().get("command_line", Type.STRING),
                ruleContext.getPrerequisite("plugin", FilesToRunProvider.class),
                ruleContext.getPrerequisite("runtime"),
                blacklistedProtos.build()))
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
