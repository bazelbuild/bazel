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
import java.util.Optional;

/**
 * A minimal context available during rule definition, for both native and starlark rules.
 *
 * <p>Encapsulates the services available for implementors of the {@link RuleDefinition} interface.
 */
public interface RuleDefinitionEnvironment {

  /** Returns the name of the tools repository. */
  String getToolsRepository();

  /**
   * Prepends the tools repository path to the given string and parses the result using {@link
   * Label#parseAbsoluteUnchecked}.
   *
   * <p>TODO(brandjon,twigg): Require override to handle repositoryMapping? Note that
   * Label.parseAbsoluteUnchecked itself is deprecated because of repositoryMapping!
   */
  default Label getToolsLabel(String labelValue) {
    return Label.parseAbsoluteUnchecked(getToolsRepository() + labelValue);
  }

  /**
   * Returns a label for network allowlist for tests if one should be added.
   *
   * <p>TODO(b/192694287): Remove once we migrate all tests from the allowlist.
   */
  default Optional<Label> getNetworkAllowlistForTests() {
    return Optional.empty();
  }
}
