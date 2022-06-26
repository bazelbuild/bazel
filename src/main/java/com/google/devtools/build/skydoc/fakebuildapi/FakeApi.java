// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi.FakeStructProviderApi;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigGlobalLibrary;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigStarlarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.repository.FakeRepositoryModule;
import com.google.devtools.build.skydoc.rendering.AspectInfoWrapper;
import com.google.devtools.build.skydoc.rendering.ProviderInfoWrapper;
import com.google.devtools.build.skydoc.rendering.RuleInfoWrapper;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import java.util.List;
import net.starlark.java.eval.Starlark;

/** Defines the fake .bzl environment. */
public final class FakeApi {

  private FakeApi() {} // uninstantiable

  /**
   * Adds the predeclared environment containing the fake build API.
   *
   * @param rules the list of {@link RuleInfo} objects, to which 'rule' and 'repository_rule'
   *     invocation information will be added
   * @param providers the list of {@link ProviderInfo} objects, to which 'provider' invocation
   *     information will be added
   * @param aspects the list of {@link AspectInfo} objects, to which 'aspect' invocation information
   *     will be added
   */
  public static void addPredeclared(
      ImmutableMap.Builder<String, Object> env,
      /* out parameters: */
      List<RuleInfoWrapper> rules,
      List<ProviderInfoWrapper> providers,
      List<AspectInfoWrapper> aspects) {

    Starlark.addMethods(env, new FakeBuildApiGlobals()); // e.g. configuration_field func
    Starlark.addMethods(
        env, new FakeStarlarkRuleFunctionsApi(rules, providers, aspects)); // e.g. rule func
    env.put("attr", new FakeStarlarkAttrModuleApi());
    env.put("struct", new FakeStructProviderApi());
    new ConfigBootstrap(
            new FakeConfigStarlarkCommon(), //
            new FakeConfigApi(),
            new FakeConfigGlobalLibrary())
        .addBindingsToBuilder(env);
    new RepositoryBootstrap(new FakeRepositoryModule(rules)).addBindingsToBuilder(env);
  }
}
