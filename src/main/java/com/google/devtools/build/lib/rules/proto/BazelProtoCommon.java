// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoCommonApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Protocol buffers support for Starlark. */
public class BazelProtoCommon implements ProtoCommonApi {
  public static final BazelProtoCommon INSTANCE = new BazelProtoCommon();

  protected BazelProtoCommon() {}

  @StarlarkMethod(
      name = "create_proto_info",
      documented = false,
      parameters = {@Param(name = "ctx", doc = "The rule context")},
      useStarlarkThread = true)
  public ProtoInfo createProtoInfo(StarlarkRuleContext ruleContext, StarlarkThread thread)
      throws EvalException {
    Label label =
        ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
            .label();
    if (!label.getPackageIdentifier().getRepository().toString().equals("@_builtins")) {
      throw Starlark.errorf("Rule in '%s' cannot use private API", label.getPackageName());
    }

    return ProtoCommon.createProtoInfo(
        ruleContext.getRuleContext(),
        ruleContext
            .getRuleContext()
            .getFragment(ProtoConfiguration.class)
            .generatedProtosInVirtualImports());
  }

  @StarlarkMethod(
      name = "write_descriptor_set",
      documented = false,
      parameters = {
        @Param(name = "ctx", doc = "The rule context"),
        @Param(name = "proto_info", doc = "The ProtoInfo")
      },
      useStarlarkThread = true)
  public void writeDescriptorSet(
      StarlarkRuleContext ruleContext, ProtoInfo protoInfo, StarlarkThread thread)
      throws EvalException {
    Label label =
        ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
            .label();
    if (!label.getPackageIdentifier().getRepository().toString().equals("@_builtins")) {
      throw Starlark.errorf("Rule in '%s' cannot use private API", label.getPackageName());
    }

    ProtoCompileActionBuilder.writeDescriptorSet(
        ruleContext.getRuleContext(), protoInfo, Services.ALLOW);
  }
}
