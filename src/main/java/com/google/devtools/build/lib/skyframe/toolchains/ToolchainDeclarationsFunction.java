// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil.InvalidTargetPatternException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.RegisteredToolchainsFunctionException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** {@link SkyFunction} that expands all registered toolchains. */
public class ToolchainDeclarationsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws RegisteredToolchainsFunctionException, InterruptedException {
    var key = (ToolchainDeclarationsValue.Key) skyKey;
    var mainRepoMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepoMapping == null) {
      return null;
    }

    var mainRepoParser =
        new TargetPattern.Parser(
            PathFragment.EMPTY_FRAGMENT, RepositoryName.MAIN, mainRepoMapping.repositoryMapping());
    var targetPatternBuilder = ImmutableList.<SignedTargetPattern>builder();

    // Get the toolchains from the configuration.
    // Reverse the list so the last one defined takes precedences.
    try {
      targetPatternBuilder.addAll(
          TargetPatternUtil.parseAllSigned(key.extraToolchains().reverse(), mainRepoParser));
    } catch (InvalidTargetPatternException e) {
      throw new RegisteredToolchainsFunctionException(
          new InvalidToolchainLabelException(e), Transience.PERSISTENT);
    }

    // Get registered toolchains from the external dep graph.
    ImmutableList<TargetPattern> bzlmodToolchains = getBzlmodToolchains(env);
    if (bzlmodToolchains == null) {
      return null;
    }
    targetPatternBuilder.addAll(TargetPatternUtil.toSigned(bzlmodToolchains));

    // Expand target patterns.
    try {
      var toolchainLabels =
          TargetPatternUtil.expandTargetPatterns(
              env, targetPatternBuilder.build(), FilteringPolicies.ruleTypeExplicit("toolchain"));
      if (env.valuesMissing()) {
        return null;
      }
      return new ToolchainDeclarationsValue(toolchainLabels.asList());
    } catch (InvalidTargetPatternException e) {
      throw new RegisteredToolchainsFunctionException(
          new InvalidToolchainLabelException(e), Transience.PERSISTENT);
    }
  }

  @Nullable
  private static ImmutableList<TargetPattern> getBzlmodToolchains(Environment env)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    var bazelDepGraphValue = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (bazelDepGraphValue == null) {
      return null;
    }
    var repoMappingKeyToModule =
        bazelDepGraphValue.getDepGraph().values().stream()
            .filter(module -> !module.getToolchainsToRegister().isEmpty())
            .collect(
                toImmutableMap(
                    module ->
                        RepositoryMappingValue.key(
                            bazelDepGraphValue
                                .getCanonicalRepoNameLookup()
                                .inverse()
                                .get(module.getKey())),
                    module -> module));
    var moduleRepoMappings = env.getValuesAndExceptions(repoMappingKeyToModule.keySet());
    if (env.valuesMissing()) {
      return null;
    }
    var toolchains = ImmutableList.<TargetPattern>builder();
    for (var repoMappingKeyAndModule : repoMappingKeyToModule.entrySet()) {
      var repoMappingKey = repoMappingKeyAndModule.getKey();
      var module = repoMappingKeyAndModule.getValue();
      var repoMappingValue = (RepositoryMappingValue) moduleRepoMappings.get(repoMappingKey);
      if (repoMappingValue == null) {
        return null;
      }
      var parser =
          new TargetPattern.Parser(
              PathFragment.EMPTY_FRAGMENT,
              repoMappingKey.repoName(),
              repoMappingValue.repositoryMapping());
      for (String pattern : module.getToolchainsToRegister()) {
        try {
          toolchains.add(parser.parse(pattern));
        } catch (TargetParsingException e) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(pattern, e), Transience.PERSISTENT);
        }
      }
    }
    return toolchains.build();
  }
}
