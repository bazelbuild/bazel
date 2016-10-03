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

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;

/**
 * An aspect that transpiles .proto dependencies using the J2ObjC proto plugin
 * and //tools/objc:standalone_protoc. N.B.: This tool has not yet been released into
 * open-source.
 */
public class BazelJ2ObjcProtoAspect extends AbstractJ2ObjcProtoAspect {
  public static final String NAME = "BazelJ2ObjcProtoAspect";

  public BazelJ2ObjcProtoAspect(String toolsRepository) {
    super(toolsRepository);
  }

  @Override
  protected AspectDefinition.Builder addAdditionalAttributes(AspectDefinition.Builder builder) {
    return builder
        .add(attr("$protoc", LABEL)
            .cfg(HOST)
            .exec()
            .singleArtifact()
            .value(Label.parseAbsoluteUnchecked(
                toolsRepository + "//tools/objc:standalone_protoc")))
        .add(attr("$j2objc_plugin", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                toolsRepository + "//third_party/java/j2objc:proto_plugin")));
  }

  @Override
  protected boolean checkShouldCreateAspect(RuleContext ruleContext) {
    return true;
  }

  @Override
  protected void createActions(
      ConfiguredTarget base,
      RuleContext ruleContext,
      Iterable<Artifact> protoSources,
      NestedSet<Artifact> transitiveProtoSources,
      Iterable<Artifact> headerMappingFiles,
      Iterable<Artifact> classMappingFiles,
      J2ObjcSource j2ObjcSource) {
    String genDir = ruleContext.getConfiguration().getGenfilesDirectory(
        ruleContext.getRule().getRepository()).getExecPathString();
    Artifact compiler = ruleContext.getPrerequisiteArtifact("$protoc", Mode.HOST);
    Artifact j2objcPlugin = ruleContext.getPrerequisiteArtifact("$j2objc_plugin", Mode.HOST);

    String langPluginParameter = Joiner.on(',').join(J2OBJC_PLUGIN_PARAMS) + ":" + genDir;
    String command = String.format(
        "%s --plugin=protoc-gen-j2objc=%s --j2objc_out=%s --proto_path=%s --proto_path=. "
            + "--absolute_paths %s",
        compiler.getExecPathString(),
        j2objcPlugin.getExecPathString(),
        langPluginParameter,
        genDir,
        Joiner.on(" ").join(Artifact.toExecPaths(protoSources)));

    ruleContext.registerAction(
        ObjcRuleClasses.spawnBashOnDarwinActionBuilder(command)
            .setMnemonic("GeneratingJ2ObjcProtos")
            .addInput(compiler)
            .addInput(j2objcPlugin)
            .addInputs(protoSources)
            .addTransitiveInputs(transitiveProtoSources)
            .addOutputs(j2ObjcSource.getObjcSrcs())
            .addOutputs(j2ObjcSource.getObjcHdrs())
            .addOutputs(headerMappingFiles)
            .addOutputs(classMappingFiles)
            .build(ruleContext));
  }
}
