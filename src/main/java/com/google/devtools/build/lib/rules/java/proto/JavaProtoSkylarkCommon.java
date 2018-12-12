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
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaProtoCommonApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A class that exposes Java common methods for proto compilation. */
public class JavaProtoSkylarkCommon
    implements JavaProtoCommonApi<Artifact, SkylarkRuleContext, ConfiguredTarget> {
  @Override
  public void createProtoCompileAction(
      SkylarkRuleContext skylarkRuleContext,
      ConfiguredTarget target,
      Artifact sourceJar,
      String protoToolchainAttr,
      String flavour)
      throws EvalException {
    ProtoInfo protoInfo = target.get(ProtoInfo.PROVIDER);
    ProtoCompileActionBuilder.registerActions(
        skylarkRuleContext.getRuleContext(),
        ImmutableList.of(
            new ProtoCompileActionBuilder.ToolchainInvocation(
                flavour,
                getProtoToolchainProvider(skylarkRuleContext, protoToolchainAttr),
                sourceJar.getExecPathString())),
        protoInfo,
        skylarkRuleContext.getLabel(),
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
      SkylarkRuleContext skylarkRuleContext, String protoToolchainAttr) throws EvalException {
    TransitiveInfoCollection runtime =
        getProtoToolchainProvider(skylarkRuleContext, protoToolchainAttr).runtime();
    return JavaInfo.Builder.create()
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaInfo.getProvider(JavaCompilationArgsProvider.class, runtime))
        .build();
  }

  @Override
  // TODO(b/78512644): migrate callers to passing explicit proto javacopts or using custom
  // toolchains, and delete
  public ImmutableList<String> getJavacOpts(
      SkylarkRuleContext skylarkRuleContext, String javaToolchainAttr) throws EvalException {
    ConfiguredTarget javaToolchainConfigTarget =
        (ConfiguredTarget) checkNotNull(skylarkRuleContext.getAttr().getValue(javaToolchainAttr));
    JavaToolchainProvider toolchain =
        checkNotNull(JavaToolchainProvider.from(javaToolchainConfigTarget));

    return ProtoJavacOpts.constructJavacOpts(skylarkRuleContext.getRuleContext(), toolchain);
  }

  private static ProtoLangToolchainProvider getProtoToolchainProvider(
      SkylarkRuleContext skylarkRuleContext, String protoToolchainAttr) throws EvalException {
    ConfiguredTarget javaliteToolchain =
        (ConfiguredTarget) checkNotNull(skylarkRuleContext.getAttr().getValue(protoToolchainAttr));
    return checkNotNull(javaliteToolchain.getProvider(ProtoLangToolchainProvider.class));
  }
}
