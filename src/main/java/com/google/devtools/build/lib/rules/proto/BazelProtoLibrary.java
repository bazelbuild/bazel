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

import static com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;

/** An implementation for the "proto_library" rule. */
public class BazelProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ImmutableList<Artifact> protoSources =
        ruleContext.getPrerequisiteArtifacts("srcs", TARGET).list();
    NestedSet<Artifact> strictImportableProtosForDependents =
        ProtoCommon.computeStrictImportableProtosForDependents(ruleContext, protoSources);
    ProtoCommon.checkSourceFilesAreInSamePackage(ruleContext);

    NestedSet<Artifact> transitiveProtoSources =
        ProtoCommon.collectTransitiveProtoSources(ruleContext, protoSources);
    NestedSet<Artifact> strictImportableProtos =
        ProtoCommon.computeStrictImportableProtos(ruleContext);

    NestedSet<Artifact> exportedProtos = ProtoCommon.computeExportedProtos(ruleContext);

    String protoSourceRoot = ProtoCommon.getProtoSourceRoot(ruleContext);
    NestedSet<String> strictImportableProtoSourceRoots =
        ProtoCommon.computeStrictImportableProtoSourceRoots(ruleContext, protoSourceRoot);
    NestedSet<String> exportedProtoSourceRoots =
        ProtoCommon.computeExportedProtoSourceRoots(ruleContext, protoSourceRoot);
    NestedSet<String> transitiveProtoSourceRoots =
        ProtoCommon.computeTransitiveProtoSourceRoots(ruleContext, protoSourceRoot);

    Artifact directDescriptorSet =
        ruleContext.getGenfilesArtifact(
            ruleContext.getLabel().getName() + "-descriptor-set.proto.bin");
    NestedSet<Artifact> dependenciesDescriptorSets =
        ProtoCommon.collectDependenciesDescriptorSets(ruleContext);
    NestedSet<Artifact> transitiveDescriptorSetOutput =
        NestedSetBuilder.fromNestedSet(dependenciesDescriptorSets).add(directDescriptorSet).build();

    ProtoSourcesProvider protoProvider =
        ProtoSourcesProvider.create(
            protoSources,
            protoSourceRoot,
            transitiveProtoSources,
            transitiveProtoSourceRoots,
            strictImportableProtosForDependents,
            strictImportableProtos,
            strictImportableProtoSourceRoots,
            exportedProtos,
            exportedProtoSourceRoots,
            directDescriptorSet,
            transitiveDescriptorSetOutput);

    ProtoCompileActionBuilder.writeDescriptorSet(
        ruleContext,
        directDescriptorSet.getExecPathString(),
        protoProvider,
        directDescriptorSet,
        Services.ALLOW,
        dependenciesDescriptorSets);

    Runfiles dataRunfiles =
        ProtoCommon.createDataRunfilesProvider(transitiveProtoSources, ruleContext)
            .addArtifact(directDescriptorSet)
            .build();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.create(STABLE_ORDER, directDescriptorSet))
        .addProvider(RunfilesProvider.withData(Runfiles.EMPTY, dataRunfiles))
        .addProvider(ProtoSourcesProvider.class, protoProvider)
        .addSkylarkTransitiveInfo(ProtoSourcesProvider.SKYLARK_NAME, protoProvider)
        .build();
  }
}
