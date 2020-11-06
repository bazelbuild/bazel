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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.TemplateVariableInfoApi;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** Provides access to make variables from the current fragments. */
@Immutable
@AutoCodec
public final class TemplateVariableInfo extends NativeInfo implements TemplateVariableInfoApi {
  /** Provider singleton constant. */
  public static final BuiltinProvider<TemplateVariableInfo> PROVIDER = new Provider();

  /** Provider for {@link TemplateVariableInfo} objects. */
  private static class Provider extends BuiltinProvider<TemplateVariableInfo>
      implements TemplateVariableInfoApi.Provider {
    private Provider() {
      super(NAME, TemplateVariableInfo.class);
    }

    @Override
    public TemplateVariableInfo templateVariableInfo(Dict<?, ?> vars, StarlarkThread thread)
        throws EvalException {
      Map<String, String> varsMap = Dict.noneableCast(vars, String.class, String.class, "vars");
      return new TemplateVariableInfo(ImmutableMap.copyOf(varsMap), thread.getCallerLocation());
    }
  }

  private final ImmutableMap<String, String> variables;

  @AutoCodec.Instantiator
  public TemplateVariableInfo(ImmutableMap<String, String> variables, Location location) {
    super(location);
    this.variables = variables;
  }

  @Override
  public BuiltinProvider<TemplateVariableInfo> getProvider() {
    return PROVIDER;
  }

  @Override
  public ImmutableMap<String, String> getVariables() {
    return variables;
  }

  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
