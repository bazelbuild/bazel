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

package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.Map;
import java.util.Set;

/**
 * A "configuration target" that asserts whether or not it matches the
 * configuration it's bound to.
 *
 * <p>This can be used, e.g., to declare a BUILD target that defines the
 * conditions which trigger a configurable attribute branch. In general,
 * this can be used to trigger for any user-configurable build behavior.
 */
@Immutable
public final class ConfigMatchingProvider implements TransitiveInfoProvider {

  private final Label label;
  private final boolean matches;
  private final Map<String, String> settingsMap;

  /**
   * @param label the build label corresponding to this matcher
   * @param settingsMap the condition settings that trigger this matcher
   * @param matches whether or not this matcher matches the configuration associated with
   *     its configured target
   */
  public ConfigMatchingProvider(Label label, Map<String, String> settingsMap,
      boolean matches) {
    this.label = label;
    this.settingsMap = settingsMap;
    this.matches = matches;
  }

  /**
   * The target's label.
   */
  public Label label() {
    return label;
  }

  /**
   * Whether or not the configuration criteria defined by this target match
   * its actual configuration.
   */
  public boolean matches() {
    return matches;
  }

  /**
   * Returns true if this matcher's conditions are a proper superset of another matcher's
   * conditions, i.e. if this matcher is a specialization of the other one.
   */
  public boolean refines(ConfigMatchingProvider other) {
    Set<Map.Entry<String, String>> settings = settingsMap.entrySet();
    Set<Map.Entry<String, String>> otherSettings = other.settingsMap.entrySet();
    return settings.containsAll(otherSettings) && settings.size() > otherSettings.size();
  }
}
