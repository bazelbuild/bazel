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
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoToolchainInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Toolchain for {@code proto_*} rules. */
@AutoValue
public abstract class ProtoToolchainInfo extends NativeInfo
    implements ProtoToolchainInfoApi<FilesToRunProvider> {
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

    return create(compiler, protoConfiguration.protocOpts());
  }

  private static final ProtoToolchainInfo create(
      FilesToRunProvider compiler, ImmutableList<String> compilerOptions) {
    Preconditions.checkNotNull(compiler);
    Preconditions.checkNotNull(compilerOptions);

    return new AutoValue_ProtoToolchainInfo(PROVIDER, compiler, compilerOptions);
  }

  @Override
  public abstract FilesToRunProvider getCompiler();

  @Override
  public abstract ImmutableList<String> getCompilerOptions();

  /** Provider class for {@link ProtoToolchainInfo} objects. */
  public static class ProtoToolchainInfoProvider extends BuiltinProvider<ProtoToolchainInfo>
      implements ProtoToolchainInfoApi.Provider<FilesToRunProvider> {
    public ProtoToolchainInfoProvider() {
      super(ProtoToolchainInfoApi.NAME, ProtoToolchainInfo.class);
    }

    @Override
    public ProtoToolchainInfoApi<FilesToRunProvider> create(
        FilesToRunProvider protoc, Sequence<?> protocOptions) throws EvalException {
      return ProtoToolchainInfo.create(
          protoc,
          Sequence.cast(protocOptions, String.class, "compiler_options").getImmutableList());
    }
  }
}
