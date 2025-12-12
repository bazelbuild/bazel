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

package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import build.stack.devtools.build.constellate.fakebuildapi.FakeStructApi.FakeStructProviderApi;
import build.stack.devtools.build.constellate.fakebuildapi.config.FakeConfigGlobalLibrary;
import build.stack.devtools.build.constellate.fakebuildapi.config.FakeConfigStarlarkCommon;
import build.stack.devtools.build.constellate.fakebuildapi.repository.FakeRepositoryModule;
import build.stack.devtools.build.constellate.rendering.AspectInfoWrapper;
import build.stack.devtools.build.constellate.rendering.MacroInfoWrapper;
import build.stack.devtools.build.constellate.rendering.ModuleExtensionInfoWrapper;
import build.stack.devtools.build.constellate.rendering.ProviderInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RepositoryRuleInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RuleInfoWrapper;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import java.util.List;
import net.starlark.java.eval.Starlark;

/** Defines the fake .bzl environment. */
public final class FakeApi {

  private FakeApi() {
  } // uninstantiable

  /**
   * Adds the predeclared environment containing the fake build API.
   *
   * @param rules     the list of {@link RuleInfo} objects, to which 'rule'
   *                  invocation information will be added
   * @param providers the list of {@link ProviderInfo} objects, to which
   *                  'provider' invocation
   *                  information will be added
   * @param aspects   the list of {@link AspectInfo} objects, to which 'aspect'
   *                  invocation information
   *                  will be added
   * @param macros    the list of {@link com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo} objects, to which 'macro'
   *                  invocation information
   *                  will be added
   * @param repositoryRules the list of {@link com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo} objects, to which 'repository_rule'
   *                  invocation information
   *                  will be added
   * @param moduleExtensions the list of {@link com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo} objects, to which 'module_extension'
   *                  invocation information
   *                  will be added
   * @param nativeRules the map of native rule name to {@link RuleInfo} for native Bazel rules
   */
  public static void addPredeclared(
      ImmutableMap.Builder<String, Object> env,
      /* out parameters: */
      List<RuleInfoWrapper> rules,
      List<ProviderInfoWrapper> providers,
      List<AspectInfoWrapper> aspects,
      List<MacroInfoWrapper> macros,
      List<RepositoryRuleInfoWrapper> repositoryRules,
      List<ModuleExtensionInfoWrapper> moduleExtensions,
      java.util.Map<String, RuleInfo> nativeRules) {

    Starlark.addMethods(env, new FakeBuildApiGlobals()); // e.g. configuration_field func
    Starlark.addMethods(
        env, new FakeStarlarkRuleFunctionsApi(rules, providers, aspects, macros)); // e.g. rule func
    env.put("attr", new FakeStarlarkAttrModuleApi());
    env.put("struct", new FakeStructProviderApi());
    env.put("native", new FakeStarlarkNativeModuleApi(nativeRules));
    new ConfigBootstrap(
        new FakeConfigStarlarkCommon(), //
        new FakeConfigApi(),
        new FakeConfigGlobalLibrary())
        .addBindingsToBuilder(env);
    new RepositoryBootstrap(new FakeRepositoryModule(rules, repositoryRules, moduleExtensions)).addBindingsToBuilder(env);
  }
}
