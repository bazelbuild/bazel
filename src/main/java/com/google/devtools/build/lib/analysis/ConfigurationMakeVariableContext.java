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
import com.google.devtools.build.lib.analysis.MakeVariableSupplier.MapBackedMakeVariableSupplier;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier.PackageBackedMakeVariableSupplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Implements make variable expansion for make variables that depend on the configuration and the
 * target (not on behavior of the {@link ConfiguredTarget} implementation). Retrieved Make variable
 * value can be modified using {@link MakeVariableSupplier}
 */
public class ConfigurationMakeVariableContext implements MakeVariableExpander.Context {

  private final ImmutableList<? extends MakeVariableSupplier> allMakeVariableSuppliers;

  // TODO(b/37567440): Remove when Skylark callers can be updated to get this from
  // CcToolchainProvider. We should use CcCommon.CC_TOOLCHAIN_ATTRIBUTE_NAME, but we didn't want to
  // pollute core with C++ specific constant.
  protected static final ImmutableList<String> DEFAULT_MAKE_VARIABLE_ATTRIBUTES =
      ImmutableList.of(":cc_toolchain", "toolchains");

  public ConfigurationMakeVariableContext(
      RuleContext ruleContext, Package pkg, BuildConfiguration configuration) {
    this(
        ruleContext.getMakeVariables(DEFAULT_MAKE_VARIABLE_ATTRIBUTES),
        pkg,
        configuration,
        ImmutableList.<MakeVariableSupplier>of());
  }

  public ConfigurationMakeVariableContext(
      ImmutableMap<String, String> ruleMakeVariables,
      Package pkg,
      BuildConfiguration configuration) {
    this(ruleMakeVariables, pkg, configuration, ImmutableList.<MakeVariableSupplier>of());
  }

  public ConfigurationMakeVariableContext(
      RuleContext ruleContext,
      Package pkg,
      BuildConfiguration configuration,
      Iterable<? extends MakeVariableSupplier> makeVariableSuppliers) {
    this(
        ruleContext.getMakeVariables(DEFAULT_MAKE_VARIABLE_ATTRIBUTES),
        pkg,
        configuration,
        makeVariableSuppliers);
  }

  public ConfigurationMakeVariableContext(
      ImmutableMap<String, String> ruleMakeVariables,
      Package pkg,
      BuildConfiguration configuration,
      Iterable<? extends MakeVariableSupplier> extraMakeVariableSuppliers) {
    this.allMakeVariableSuppliers =
        ImmutableList.<MakeVariableSupplier>builder()
            .addAll(Preconditions.checkNotNull(extraMakeVariableSuppliers))
            .add(new MapBackedMakeVariableSupplier(ruleMakeVariables))
            .add(new MapBackedMakeVariableSupplier(configuration.getCommandLineBuildVariables()))
            .add(new PackageBackedMakeVariableSupplier(pkg, configuration.getPlatformName()))
            .add(new MapBackedMakeVariableSupplier(configuration.getGlobalMakeEnvironment()))
            .build();
  }

  @Override
  public String lookupMakeVariable(String variableName) throws ExpansionException {
    for (MakeVariableSupplier supplier : allMakeVariableSuppliers) {
      String variableValue = supplier.getMakeVariable(variableName);
      if (variableValue != null) {
        return variableValue;
      }
    }
    throw new MakeVariableExpander.ExpansionException("$(" + variableName + ") not defined");
  }

  public SkylarkDict<String, String> collectMakeVariables() {
    Map<String, String> map = new LinkedHashMap<>();
    // Collect variables in the reverse order as in lookupMakeVariable
    // because each update is overwriting.
    for (MakeVariableSupplier supplier : allMakeVariableSuppliers.reverse()) {
      map.putAll(supplier.getAllMakeVariables());
    }
    return SkylarkDict.<String, String>copyOf(null, map);
  }
}
