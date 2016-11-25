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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSupportDataProvider;
import com.google.devtools.build.lib.rules.proto.SupportData;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A class that exposes Java common methods for proto compilation.
 */
@SkylarkModule(name = "java_proto_common", doc = "Helper class for Java proto compilation.")
public class JavaProtoSkylarkCommon {
  @SkylarkCallable(
      name = "create_java_lite_proto_compile_action",
      // This function is experimental for now.
      documented = false,
      // There's 2 mandatory positional arguments, the Skylark context and the ConfiguredTarget.
      mandatoryPositionals = 2,
      parameters = {
          @Param(
              name = "src_jar",
              positional = false,
              named = true,
              type = Artifact.class
          ),
          @Param(
              name = "proto_toolchain_attr",
              positional = false,
              named = true,
              type = String.class
          )
      }
  )
  public static void createProtoCompileAction(
      SkylarkRuleContext skylarkRuleContext,
      ConfiguredTarget target,
      Artifact sourceJar,
      String protoToolchainAttr) {
    SupportData supportData =
        checkNotNull(target.getProvider(ProtoSupportDataProvider.class).getSupportData());
    ProtoCompileActionBuilder.registerActions(
        skylarkRuleContext.getRuleContext(),
        ImmutableList.of(
            new ProtoCompileActionBuilder.ToolchainInvocation(
                "javalite",
                getProtoToolchainProvider(skylarkRuleContext, protoToolchainAttr),
                sourceJar.getExecPathString())),
        supportData.getDirectProtoSources(),
        supportData.getTransitiveImports(),
        supportData.getProtosInDirectDeps(),
        ImmutableList.of(sourceJar),
        "JavaLite",
        true /* allowServices */);
  }

  private static ProtoLangToolchainProvider getProtoToolchainProvider(
      SkylarkRuleContext skylarkRuleContext, String protoToolchainAttr) {
    ConfiguredTarget javaliteToolchain = (ConfiguredTarget) checkNotNull(
            skylarkRuleContext.getAttr().getValue(protoToolchainAttr));
    return checkNotNull(javaliteToolchain.getProvider(ProtoLangToolchainProvider.class));
  }
}
