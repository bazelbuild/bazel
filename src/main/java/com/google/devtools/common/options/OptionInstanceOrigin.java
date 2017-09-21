// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import javax.annotation.Nullable;

/**
 * Contains metadata describing the origin of an option. This includes its priority, a message about
 * where it came from, and whether it was set explicitly or expanded/implied by other flags.
 */
public class OptionInstanceOrigin {
  private final OptionPriority priority;
  @Nullable private final String source;
  @Nullable private final OptionDefinition implicitDependent;
  @Nullable private final OptionDefinition expandedFrom;

  public OptionInstanceOrigin(
      OptionPriority priority,
      String source,
      OptionDefinition implicitDependent,
      OptionDefinition expandedFrom) {
    this.priority = priority;
    this.source = source;
    this.implicitDependent = implicitDependent;
    this.expandedFrom = expandedFrom;
  }

  public OptionPriority getPriority() {
    return priority;
  }

  @Nullable
  public String getSource() {
    return source;
  }

  @Nullable
  public OptionDefinition getImplicitDependent() {
    return implicitDependent;
  }

  @Nullable
  public OptionDefinition getExpandedFrom() {
    return expandedFrom;
  }
}
