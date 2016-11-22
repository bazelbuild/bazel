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

package com.google.devtools.build.lib.rules;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.TreeMap;

/**
 * A toolchain, determined from the current platform.
 */
@Immutable
public final class ToolchainProvider implements TransitiveInfoProvider {
  private final ImmutableMap<String, String> makeVariables;

  public ToolchainProvider(ImmutableMap<String, String> makeVariables) {
    this.makeVariables = makeVariables;
  }

  public ImmutableMap<String, String> getMakeVariables() {
    return makeVariables;
  }

  public static ImmutableMap<String, String> getToolchainMakeVariables(
      RuleContext ruleContext, String attributeName) {
    // Cannot be an ImmutableMap.Builder because we want to support duplicate keys
    TreeMap<String, String> result = new TreeMap<>();
    for (ToolchainProvider provider :
        ruleContext.getPrerequisites(attributeName, Mode.TARGET, ToolchainProvider.class)) {
      result.putAll(provider.getMakeVariables());
    }

    return ImmutableMap.copyOf(result);
  }
}
