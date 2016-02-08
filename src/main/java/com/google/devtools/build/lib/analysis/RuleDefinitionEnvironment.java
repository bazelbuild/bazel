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

import com.google.devtools.build.lib.cmdline.Label;

/**
 * Encapsulates the services available for implementors of the {@link RuleDefinition}
 * interface.
 */
public interface RuleDefinitionEnvironment {
  /**
   * Parses the given string as a label and returns the label, by calling {@link
   * Label#parseAbsolute}. Throws a {@link IllegalArgumentException} if the parsing fails.
   */
  Label getLabel(String labelValue);

  /**
   * Prepends the tools repository path to the given string and parses the result
   * using {@link RuleDefinitionEnvironment#getLabel}
   */
  Label getToolsLabel(String labelValue);
}
