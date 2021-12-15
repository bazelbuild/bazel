// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoBootstrap;
import javax.annotation.Nullable;

/** Toolchain for {@code proto_*} rules. */
@AutoValue
public abstract class ProtoToolchainInfo extends NativeInfo {
  public static final ProtoToolchainInfoProvider PROVIDER = new ProtoToolchainInfoProvider();

  /** Creates a {@link ProtoToolchainInfo} from a {@link RuleContext}. */
  @Nullable
  public static ProtoToolchainInfo fromRuleContext(RuleContext ruleContext) {
    Preconditions.checkNotNull(ruleContext);

    FilesToRunProvider compiler = ruleContext.getExecutablePrerequisite(":proto_compiler");
    if (compiler == null) {
      return null;
    }

    ProtoConfiguration protoConfiguration = ruleContext.getFragment(ProtoConfiguration.class);
    if (protoConfiguration == null) {
      return null;
    }

    return new AutoValue_ProtoToolchainInfo(PROVIDER, compiler, protoConfiguration.protocOpts());
  }

  public abstract FilesToRunProvider getCompiler();

  public abstract ImmutableList<String> getCompilerOptions();

  /** Provider class for {@link ProtoToolchainInfo} objects. */
  public static class ProtoToolchainInfoProvider extends BuiltinProvider<ProtoToolchainInfo> {
    public ProtoToolchainInfoProvider() {
      super(ProtoBootstrap.PROTO_TOOLCHAIN_INFO_STARLARK_NAME, ProtoToolchainInfo.class);
    }
  }
}
