// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkbuildapi.FragmentCollectionApi;
import com.google.devtools.build.lib.syntax.EvalException;
import javax.annotation.Nullable;

/** Represents a collection of configuration fragments in Skylark. */
// Documentation can be found at ctx.fragments
@Immutable
public class FragmentCollection implements FragmentCollectionApi {
  private final RuleContext ruleContext;
  private final ConfigurationTransition transition;

  public FragmentCollection(RuleContext ruleContext, ConfigurationTransition transition) {
    this.ruleContext = ruleContext;
    this.transition = transition;
  }

  @Override
  @Nullable
  public Object getValue(String name) throws EvalException {
    return ruleContext.getSkylarkFragment(name, transition);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ruleContext.getSkylarkFragmentNames(transition);
  }

  @Override
  @Nullable
  public String getErrorMessageForUnknownField(String name) {
    return String.format(
        "There is no configuration fragment named '%s' in %s configuration. "
        + "Available fragments: %s",
        name, getConfigurationName(transition), fieldsToString());
  }

  private String fieldsToString() {
    return String.format("'%s'", Joiner.on("', '").join(getFieldNames()));
  }

  public static String getConfigurationName(ConfigurationTransition config) {
    return config.isHostTransition() ? "host" : "target";
  }

  @Override
  public String toString() {
    return getConfigurationName(transition) + ": [ " + fieldsToString() + "]";
  }
}
