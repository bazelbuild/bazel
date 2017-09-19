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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;

/**
 * Container for an attribute name and access mode such as are commonly passed to
 * {@link RuleContext#getPrerequisite(String, Mode)} and related methods.
 */
public class Attribute {
  private final String name;
  private final Mode accessMode;

  /**
   * Creates an attribute reference with the given name and access mode.
   */
  public Attribute(String name, Mode accessMode) {
    this.name = name;
    this.accessMode = accessMode;
  }

  public String getName() {
    return name;
  }

  public Mode getAccessMode() {
    return accessMode;
  }
}
