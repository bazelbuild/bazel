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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.ClassObject;

import javax.annotation.Nullable;

/**
 * Represents a collection of configuration fragments in Skylark.
 */
// Documentation can be found at ctx.fragments
@Immutable
@SkylarkModule(name = "fragments", doc = "Possible fields are "
    + "<a href=\"apple.html\">apple</a>, <a href=\"cpp.html\">cpp</a>, "
    + "<a href=\"java.html\">java</a> and <a href=\"jvm.html\">jvm</a>. "
    + "Access a specific fragment by its field name ex:</p><code>ctx.fragments.apple</code></p>"
    + "Note that rules have to declare their required fragments in order to access them "
    + "(see <a href=\"../rules.html#fragments\">here</a>).")
public class FragmentCollection implements ClassObject {
  private final RuleContext ruleContext;
  private final ConfigurationTransition config;

  public FragmentCollection(RuleContext ruleContext, ConfigurationTransition config) {
    this.ruleContext = ruleContext;
    this.config = config;
  }

  @Override
  @Nullable
  public Object getValue(String name) {
    return ruleContext.getSkylarkFragment(name, config);
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return ruleContext.getSkylarkFragmentNames(config);
  }

  @Override
  @Nullable
  public String errorMessage(String name) {
    return String.format(
        "There is no configuration fragment named '%s' in %s configuration. "
        + "Available fragments: %s",
        name, getConfigurationName(config), printKeys());
  }

  private String printKeys() {
    return String.format("'%s'", Joiner.on("', '").join(getKeys()));
  }

  public static String getConfigurationName(ConfigurationTransition config) {
    return (config == ConfigurationTransition.HOST) ? "host" : "target";
  }

  @Override
  public String toString() {
    return getConfigurationName(config) + ": [ " + printKeys() + "]";
  }
}
