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

import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.rules.proto.ProtoCommon.areDepsStrict;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/** An implementation for the "proto_library" rule. */
public class BazelProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ImmutableList<Artifact> protoSources =
        ruleContext.getPrerequisiteArtifacts("srcs", TARGET).list();
    NestedSet<Artifact> checkDepsProtoSources =
        ProtoCommon.getCheckDepsProtoSources(ruleContext, protoSources);
    ProtoCommon.checkSourceFilesAreInSamePackage(ruleContext);

    NestedSet<Artifact> transitiveImports =
        ProtoCommon.collectTransitiveImports(ruleContext, protoSources);

    NestedSet<Artifact> protosInDirectDeps =
        areDepsStrict(ruleContext) ? ProtoCommon.computeProtosInDirectDeps(ruleContext) : null;

    final SupportData supportData =
        SupportData.create(
            Predicates.<TransitiveInfoCollection>alwaysTrue() /* nonWeakDepsPredicate */,
            protoSources,
            protosInDirectDeps,
            transitiveImports,
            !protoSources.isEmpty());

    Runfiles.Builder dataRunfiles =
        ProtoCommon.createDataRunfilesProvider(transitiveImports, ruleContext);

    RuleConfiguredTargetBuilder result = new RuleConfiguredTargetBuilder(ruleContext);

    Artifact descriptorSetOutput = null;
    if (checkDepsProtoSources.isEmpty() || !outputDescriptorSetFlagEnabled(ruleContext)) {
      result.setFilesToBuild(NestedSetBuilder.<Artifact>create(STABLE_ORDER));
    } else {
      descriptorSetOutput =
          ruleContext.getGenfilesArtifact(
              ruleContext.getLabel().getName() + "-descriptor-set.proto.bin");
      ProtoCompileActionBuilder.writeDescriptorSet(
          ruleContext,
          descriptorSetOutput.getExecPathString(),
          checkDepsProtoSources,
          transitiveImports,
          protosInDirectDeps,
          ImmutableList.of(descriptorSetOutput),
          true /* allowServices */);

      dataRunfiles.addArtifact(descriptorSetOutput);

      result.setFilesToBuild(NestedSetBuilder.create(STABLE_ORDER, descriptorSetOutput));
    }

    // TODO(bazel-team): this second constructor argument is superfluous and should be removed.
    ProtoSourcesProvider sourcesProvider =
        ProtoSourcesProvider.create(
            transitiveImports,
            transitiveImports,
            protoSources,
            checkDepsProtoSources,
            descriptorSetOutput);

    return result
        .addProvider(RunfilesProvider.withData(Runfiles.EMPTY, dataRunfiles.build()))
        .addProvider(ProtoSourcesProvider.class, sourcesProvider)
        .addProvider(ProtoSupportDataProvider.class, new ProtoSupportDataProvider(supportData))
        .addSkylarkTransitiveInfo(ProtoSourcesProvider.SKYLARK_NAME, sourcesProvider)
        .build();
  }

  private boolean outputDescriptorSetFlagEnabled(RuleContext ruleContext) {
    return ruleContext
        .getConfiguration()
        .getFragment(ProtoConfiguration.class)
        .outputDescriptorSet();
  }
}
