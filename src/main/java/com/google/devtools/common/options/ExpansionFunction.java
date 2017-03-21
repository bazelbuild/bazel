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

/**
 * A function from an option parser's static setup (what flags it knows about) to an expansion
 * String[] to use for one of its options.
 */
public interface ExpansionFunction {

  /**
   * Compute the expansion for an option. May be called at any time during or after the {@link
   * OptionsParser}'s construction, or not at all.
   *
   * @param optionsData the parser's indexed information about its own options, before expansion
   *     information is computed
   * @return An expansion to use for all occurrences of this option in this parser
   */
  public String[] getExpansion(IsolatedOptionsData optionsData);
}
