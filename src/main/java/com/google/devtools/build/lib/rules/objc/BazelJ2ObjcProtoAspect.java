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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * An aspect that transpiles .proto dependencies using the J2ObjC proto plugin,
 * //tools/objc:compile_protos and //tools/objc:proto_support. N.B.: These two tools
 * have not yet been released into open-source.
 */
public class BazelJ2ObjcProtoAspect extends AbstractJ2ObjcProtoAspect {

  @Override
  protected AspectDefinition.Builder addAdditionalAttributes(AspectDefinition.Builder builder) {
    return builder
        .add(attr("$protoc_darwin", LABEL)
            .cfg(HOST)
            .exec()
            .value(parseLabel(Constants.TOOLS_REPOSITORY + "//tools/objc:compile_protos")))
        .add(attr("$protoc_support_darwin", LABEL)
            .cfg(HOST)
            .exec()
            .value(parseLabel(Constants.TOOLS_REPOSITORY + "//tools/objc:proto_support")))
        .add(attr("$j2objc_plugin", LABEL)
            .cfg(HOST)
            .exec()
            .value(parseLabel(
                "//third_party/java/j2objc:proto_plugin")));
  }

    @Override
    protected boolean checkShouldCreateAspect(RuleContext ruleContext) {
        return true;
    }

    protected void createActions(ConfiguredTarget base, RuleContext ruleContext,
      Iterable<Artifact> protoSources, NestedSet<Artifact> transitiveProtoSources,
      Iterable<Artifact> headerMappingFiles, Iterable<Artifact> classMappingFiles,
      J2ObjcSource j2ObjcSource) {
    String genDir = ruleContext.getConfiguration().getGenfilesDirectory().getExecPathString();
    Artifact compiler = ruleContext.getPrerequisiteArtifact(
        "$protoc_darwin", Mode.HOST);
    Artifact j2objcPlugin = ruleContext.getPrerequisiteArtifact("$j2objc_plugin", Mode.HOST);

    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("TranslatingJ2ObjcProtos")
        .addInput(compiler)
        .addInput(j2objcPlugin)
        .addInputs(ruleContext.getPrerequisiteArtifacts(
            "$protoc_support_darwin", Mode.HOST).list())
        .addInputs(protoSources)
        .addTransitiveInputs(transitiveProtoSources)
        .addOutputs(j2ObjcSource.getObjcSrcs())
        .addOutputs(j2ObjcSource.getObjcHdrs())
        .addOutputs(headerMappingFiles)
        .addOutputs(classMappingFiles)
        .setExecutable(new PathFragment("/usr/bin/python"))
        .setCommandLine(new CustomCommandLine.Builder()
            .add(compiler.getPath().toString())
            .add("-w")
            .add(".")  // Actions are always run with the exec root as cwd
            .add("--generate-j2objc")
            .add("--generator-param=file_dir_mapping")
            .add("--generator-param=generate_class_mappings")
            .add("--j2objc-plugin=" + j2objcPlugin.getExecPathString())
            .add("--output-dir=" + genDir)
            .addExecPaths(protoSources)
            .build())
        .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
        .build(ruleContext));
  }

    @Override
  protected boolean checkShouldCreateSources(RuleContext ruleContext) {
      return true;
  }
}
