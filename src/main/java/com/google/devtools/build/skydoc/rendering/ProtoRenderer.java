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

package com.google.devtools.build.skydoc.rendering;

import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.UserDefinedFunctionInfo;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;

/** Produces skydoc output in proto form. */
public class ProtoRenderer {

  private final ModuleInfo.Builder moduleInfo;

  public ProtoRenderer() {
    this.moduleInfo = ModuleInfo.newBuilder();
  }

  /** Appends {@link RuleInfo} protos to a {@link ModuleInfo.Builder}. */
  public ProtoRenderer appendRuleInfos(Collection<RuleInfo> ruleInfos) {
    for (RuleInfo ruleInfo : ruleInfos) {
      moduleInfo.addRuleInfo(ruleInfo);
    }
    return this;
  }

  /** Appends {@link ProviderInfo} protos to a {@link ModuleInfo.Builder}. */
  public ProtoRenderer appendProviderInfos(Collection<ProviderInfo> providerInfos) {
    for (ProviderInfo providerInfo : providerInfos) {
      moduleInfo.addProviderInfo(providerInfo);
    }
    return this;
  }

  /** Appends {@link UserDefinedFunctionInfo} protos to a {@link ModuleInfo.Builder}. */
  public ProtoRenderer appendUserDefinedFunctionInfos(Map<String, UserDefinedFunction> funcInfosMap)
      throws DocstringParseException {
    for (Map.Entry<String, UserDefinedFunction> entry : funcInfosMap.entrySet()) {
      UserDefinedFunctionInfo funcInfo =
          FunctionUtil.fromNameAndFunction(entry.getKey(), entry.getValue());
      moduleInfo.addFuncInfo(funcInfo);
    }
    return this;
  }

  /** Outputs the raw form of a {@link ModuleInfo} proto. */
  public void writeModuleInfo(BufferedOutputStream outputStream) throws IOException {
    ModuleInfo build = moduleInfo.build();
    build.writeTo(outputStream);
  }

  public ModuleInfo.Builder getModuleInfo() {
    return moduleInfo;
  }
}
