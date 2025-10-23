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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Optional;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** Utility methods for Objc rules in Starlark Builtins */
@StarlarkBuiltin(name = "objc_internal", category = DocCategory.BUILTIN, documented = false)
public class ObjcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "objc_internal";

  @StarlarkMethod(
      name = "get_apple_config",
      documented = false,
      parameters = {@Param(name = "build_config", named = true)})
  public AppleConfiguration getAppleConfig(BuildConfigurationValue buildConfiguration)
      throws EvalException {
    return buildConfiguration.getFragment(AppleConfiguration.class);
  }

  @StarlarkMethod(
      name = "get_split_build_configs",
      documented = false,
      parameters = {@Param(name = "ctx", positional = true, named = true)})
  public Dict<String, BuildConfigurationValue> getSplitBuildConfigs(
      StarlarkRuleContext starlarkRuleContext) throws EvalException {
    Map<Optional<String>, List<ConfiguredTargetAndData>> ctads =
        starlarkRuleContext
            .getRuleContext()
            .getRulePrerequisitesCollection()
            .getSplitPrerequisites(ObjcRuleClasses.CHILD_CONFIG_ATTR);
    Dict.Builder<String, BuildConfigurationValue> result = Dict.builder();
    for (Optional<String> splitTransitionKey : ctads.keySet()) {
      if (!splitTransitionKey.isPresent()) {
        throw new EvalException("unexpected empty key in split transition");
      }
      result.put(
          splitTransitionKey.get(),
          Iterables.getOnlyElement(ctads.get(splitTransitionKey)).getConfiguration());
    }
    return result.buildImmutable();
  }
}
