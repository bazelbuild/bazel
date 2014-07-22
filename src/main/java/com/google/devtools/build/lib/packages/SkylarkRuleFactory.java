// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.vfs.Path;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * A 'storage class' for Skylark rule classes.
 */
public class SkylarkRuleFactory {

  /**
   * Dynamically loaded Skylark rule classes. The key of the outer map is the file
   * where the build extension is located, the inner key is the simple name of the
   * rule class (e.g. "foo_binary").  
   */
  private final Map<Path, Map<String, RuleClass>> ruleClassMap;
  private final RuleClassProvider ruleClassProvider;

  @VisibleForTesting
  public SkylarkRuleFactory() {
    ruleClassMap = new HashMap<>();
    ruleClassProvider = null;
  }

  public SkylarkRuleFactory(RuleClassProvider ruleClassProvider) {
    this.ruleClassMap = new HashMap<>();
    this.ruleClassProvider = ruleClassProvider;
  }

  /**
   * Returns the (immutable, unordered) set of names of the Skylark rule classes.
   */
  public Set<String> getRuleClassNames(Path file) {
    if (ruleClassMap.containsKey(file)) {
      return ImmutableSet.copyOf(ruleClassMap.get(file).keySet());
    } else {
      return ImmutableSet.<String>of();
    }
  }

  public void clear(Path file) {
    ruleClassMap.put(file, new HashMap<String, RuleClass>());
  }

  /**
   * Adds a rule class with the given extension file to the rule factory.
   */
  public void addSkylarkRuleClass(RuleClass ruleClass, Path file) {
    Preconditions.checkState(ruleClassMap.containsKey(file),
        "Extension file cache '%s' needs to be cleared", file);
    ruleClassMap.get(file).put(ruleClass.getName(), ruleClass);
  }

  /**
   * Returns the rule class of the given name and extension file.
   */
  public RuleClass getRuleClass(String name, Path file) {
    Preconditions.checkArgument(ruleClassMap.containsKey(file),
        "Extension file cache '%s' needs to be cleared", file);
    return ruleClassMap.get(file).get(name);
  }

  /**
   * Returns true if there's a Skylark rule with the given name and extension file.
   */
  public boolean hasRuleClass(String name, Path file) {
    return ruleClassMap.containsKey(file) && ruleClassMap.get(file).containsKey(name);
  }

  /**
   * Returns a Skylark Environment for rule creation using this SkylarkRuleFactory.
   */
  public SkylarkEnvironment getSkylarkRuleClassEnvironment() {
    return ruleClassProvider.getSkylarkRuleClassEnvironment();
  }
}
