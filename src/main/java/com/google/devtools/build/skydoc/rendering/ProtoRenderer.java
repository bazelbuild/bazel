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

import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import net.starlark.java.eval.StarlarkFunction;

/** Produces skydoc output in proto form. */
public class ProtoRenderer {

  private final ModuleInfo.Builder moduleInfo;

  public ProtoRenderer() {
    this.moduleInfo = ModuleInfo.newBuilder();
  }

  /** Appends {@link RuleInfo} protos to a {@link ModuleInfo.Builder}. */
  @CanIgnoreReturnValue
  public ProtoRenderer appendRuleInfos(Collection<RuleInfo> ruleInfos) {
    for (RuleInfo ruleInfo : ruleInfos) {
      moduleInfo.addRuleInfo(ruleInfo);
    }
    return this;
  }

  /** Appends {@link ProviderInfo} protos to a {@link ModuleInfo.Builder}. */
  @CanIgnoreReturnValue
  public ProtoRenderer appendProviderInfos(Collection<ProviderInfo> providerInfos) {
    for (ProviderInfo providerInfo : providerInfos) {
      moduleInfo.addProviderInfo(providerInfo);
    }
    return this;
  }

  /** Appends {@link StarlarkFunctionInfo} protos to a {@link ModuleInfo.Builder}. */
  @CanIgnoreReturnValue
  public ProtoRenderer appendStarlarkFunctionInfos(Map<String, StarlarkFunction> funcInfosMap)
      throws DocstringParseException {
    for (Map.Entry<String, StarlarkFunction> entry : funcInfosMap.entrySet()) {
      StarlarkFunctionInfo funcInfo =
          StarlarkFunctionInfoExtractor.fromNameAndFunction(
              entry.getKey(), entry.getValue(), /* withOriginKey= */ false, LabelRenderer.DEFAULT);
      moduleInfo.addFuncInfo(funcInfo);
    }
    return this;
  }

  /** Appends module docstring protos to a {@link ModuleInfo.Builder}. */
  @CanIgnoreReturnValue
  public ProtoRenderer setModuleDocstring(String moduleDoc) {
    moduleInfo.setModuleDocstring(moduleDoc);
    return this;
  }

  /** Outputs the raw form of a {@link ModuleInfo} proto. */
  public void writeModuleInfo(BufferedOutputStream outputStream) throws IOException {
    ModuleInfo build = moduleInfo.build();
    build.writeTo(outputStream);
  }

  /** Appends {@link AspectInfo} protos to a {@link ModuleInfo.Builder}. */
  @CanIgnoreReturnValue
  public ProtoRenderer appendAspectInfos(Collection<AspectInfo> aspectInfos) {
    for (AspectInfo aspectInfo : aspectInfos) {
      moduleInfo.addAspectInfo(aspectInfo);
    }
    return this;
  }

  public ModuleInfo.Builder getModuleInfo() {
    return moduleInfo;
  }
}
