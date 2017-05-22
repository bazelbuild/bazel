// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Implements make variable expansion for make variables that depend on the
 * configuration and the target (not on behavior of the
 * {@link ConfiguredTarget} implementation)
 */
public class ConfigurationMakeVariableContext implements MakeVariableExpander.Context {
  private final Package pkg;
  private final ImmutableMap<String, String> ruleEnv;
  private final ImmutableMap<String, String> commandLineEnv;
  private final ImmutableMap<String, String> globalEnv;
  private final String platform;

  // TODO(b/37567440): Remove when Skylark callers can be updated to get this from
  // CcToolchainProvider. We should use CcCommon.CC_TOOLCHAIN_ATTRIBUTE_NAME, but we didn't want to
  // pollute core with C++ specific constant.
  private static final ImmutableList<String> defaultMakeVariableAttributes =
      ImmutableList.of(":cc_toolchain");

  public ConfigurationMakeVariableContext(
      RuleContext ruleContext, Package pkg, BuildConfiguration configuration) {
    this(ruleContext.getMakeVariables(defaultMakeVariableAttributes), pkg, configuration);
  }

  public ConfigurationMakeVariableContext(
      ImmutableMap<String, String> ruleMakeVariables,
      Package pkg,
      BuildConfiguration configuration) {
    this.pkg = pkg;
    this.ruleEnv = ruleMakeVariables;
    this.commandLineEnv = ImmutableMap.copyOf(configuration.getCommandLineBuildVariables());
    this.globalEnv = ImmutableMap.copyOf(configuration.getGlobalMakeEnvironment());
    this.platform = configuration.getPlatformName();
  }

  @Override
  public String lookupMakeVariable(String var) throws ExpansionException {
    String value = ruleEnv.get(var);
    if (value == null) {
      value = commandLineEnv.get(var);
    }
    if (value == null) {
      value = pkg.lookupMakeVariable(var, platform);
    }
    if (value == null) {
      value = globalEnv.get(var);
    }
    if (value == null) {
      throw new MakeVariableExpander.ExpansionException("$(" + var + ") not defined");
    }

    return value;
  }

  public SkylarkDict<String, String> collectMakeVariables() {
    Map<String, String> map = new LinkedHashMap<>();
    // Collect variables in the reverse order as in lookupMakeVariable
    // because each update is overwriting.
    map.putAll(pkg.getAllMakeVariables(platform));
    map.putAll(globalEnv);
    map.putAll(commandLineEnv);
    map.putAll(ruleEnv);
    return SkylarkDict.<String, String>copyOf(null, map);
  }
}
