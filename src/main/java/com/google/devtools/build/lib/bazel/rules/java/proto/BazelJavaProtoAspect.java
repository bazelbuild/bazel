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

package com.google.devtools.build.lib.bazel.rules.java.proto;

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.rules.java.proto.JavaProtoAspect;
import com.google.devtools.build.lib.rules.java.proto.RpcSupport;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import java.util.List;

/** An Aspect which BazelJavaProtoLibrary injects to build Java SPEED protos. */
public class BazelJavaProtoAspect extends JavaProtoAspect {

  public BazelJavaProtoAspect(RuleDefinitionEnvironment env) {
    super(
        BazelJavaSemantics.INSTANCE,
        new NoopRpcSupport(),
        "@com_google_protobuf//:java_toolchain",
        env);
  }

  private static class NoopRpcSupport
      implements RpcSupport {
    @Override
    public List<ProtoCompileActionBuilder.ToolchainInvocation> getToolchainInvocation(
        RuleContext ruleContext, Artifact sourceJar) {
      return ImmutableList.of();
    }

    @Override
    public boolean allowServices(RuleContext ruleContext) {
      return true;
    }

    @Override
    public NestedSet<Artifact> getForbiddenProtos(RuleContext ruleContext) {
      return NestedSetBuilder.emptySet(STABLE_ORDER);
    }

    @Override
    public ImmutableList<TransitiveInfoCollection> getRuntimes(RuleContext ruleContext) {
      return ImmutableList.of();
    }

    @Override
    public void mutateAspectDefinition(
        AspectDefinition.Builder def, AspectParameters aspectParameters) {
      // Intentionally left empty.
    }

    @Override
    public boolean checkAttributes(RuleContext ruleContext, AspectParameters aspectParameters) {
      // Intentionally left empty.
      return true;
    }
  }
}
