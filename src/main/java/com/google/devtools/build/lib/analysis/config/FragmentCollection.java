// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.SkylarkModule;

import javax.annotation.Nullable;

/**
 * Represents a collection of configuration fragments in Skylark.
 */
@Immutable
@SkylarkModule(name = "fragments", doc = "Allows access to configuration fragments.")
public class FragmentCollection implements ClassObject {
  private final RuleContext ruleContext;

  public FragmentCollection(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  @Override
  @Nullable
  public Object getValue(String name) {
    return ruleContext.getSkylarkFragment(name);
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return ruleContext.getSkylarkFragmentNames();
  }

  @Override
  @Nullable
  public String errorMessage(String name) {
    return String.format("There is no configuration fragment named '%s'. Available fragments: %s",
        name, printKeys());
  }

  private String printKeys() {
    return String.format("'%s'", Joiner.on("', '").join(getKeys()));
  }
}