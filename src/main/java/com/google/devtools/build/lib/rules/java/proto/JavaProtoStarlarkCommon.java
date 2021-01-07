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
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaProtoCommonApi;
import net.starlark.java.eval.EvalException;

/** A class that exposes Java common methods for proto compilation. */
public class JavaProtoStarlarkCommon
    implements JavaProtoCommonApi<
        Artifact, ConstraintValueInfo, StarlarkRuleContext, ConfiguredTarget> {
  @Override
  public void createProtoCompileAction(
      StarlarkRuleContext starlarkRuleContext,
      ConfiguredTarget target,
      Artifact sourceJar,
      String protoToolchainAttr,
      String flavour)
      throws EvalException {
    ProtoInfo protoInfo = target.get(ProtoInfo.PROVIDER);
    ProtoCompileActionBuilder.registerActions(
        starlarkRuleContext.getRuleContext(),
        ImmutableList.of(
            new ProtoCompileActionBuilder.ToolchainInvocation(
                flavour,
                getProtoToolchainProvider(starlarkRuleContext, protoToolchainAttr),
                sourceJar.getExecPathString())),
        protoInfo,
        starlarkRuleContext.getLabel(),
        ImmutableList.of(sourceJar),
        "JavaLite",
        Exports.DO_NOT_USE,
        Services.ALLOW);
  }

  @Override
  public boolean hasProtoSources(ConfiguredTarget target) {
    return !target.get(ProtoInfo.PROVIDER).getDirectProtoSources().isEmpty();
  }

  @Override
  public JavaInfo getRuntimeToolchainProvider(
      StarlarkRuleContext starlarkRuleContext, String protoToolchainAttr) throws EvalException {
    TransitiveInfoCollection runtime =
        getProtoToolchainProvider(starlarkRuleContext, protoToolchainAttr).runtime();
    return JavaInfo.Builder.create()
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaInfo.getProvider(JavaCompilationArgsProvider.class, runtime))
        .build();
  }

  private static ProtoLangToolchainProvider getProtoToolchainProvider(
      StarlarkRuleContext starlarkRuleContext, String protoToolchainAttr) throws EvalException {
    ConfiguredTarget javaliteToolchain =
        (ConfiguredTarget) checkNotNull(starlarkRuleContext.getAttr().getValue(protoToolchainAttr));
    return checkNotNull(javaliteToolchain.getProvider(ProtoLangToolchainProvider.class));
  }
}
