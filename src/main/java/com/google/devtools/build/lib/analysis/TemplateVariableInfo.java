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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.TemplateVariableInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.util.Map;

/** Provides access to make variables from the current fragments. */
@Immutable
@AutoCodec
public final class TemplateVariableInfo extends NativeInfo implements TemplateVariableInfoApi {
  public static final String SKYLARK_NAME = "TemplateVariableInfo";

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(/*numMandatory=*/ 1, "vars"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.<SkylarkType>of(SkylarkType.DICT));

  public static final NativeProvider<TemplateVariableInfo> PROVIDER =
      new NativeProvider<TemplateVariableInfo>(
          TemplateVariableInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        protected TemplateVariableInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) throws EvalException {
          Map<String, String> vars =
              SkylarkDict.castSkylarkDictOrNoneToDict(args[0], String.class, String.class, "vars");
          return new TemplateVariableInfo(ImmutableMap.copyOf(vars), location);
        }
      };

  private final ImmutableMap<String, String> variables;

  @AutoCodec.Instantiator
  public TemplateVariableInfo(ImmutableMap<String, String> variables, Location location) {
    super(PROVIDER, ImmutableMap.of(), location);
    this.variables = variables;
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
