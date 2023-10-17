// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.Label;

/** This class provides constants associated with the function split transition allowlist. */
public class FunctionSplitTransitionAllowlist {
  public static final String NAME = "function_transition";
  public static final String ATTRIBUTE_NAME = "$allowlist_function_transition";
  public static final String LABEL_STR = "//tools/allowlists/function_transition_allowlist";
  public static final Label LABEL = Label.parseCanonicalUnchecked(LABEL_STR);

  private FunctionSplitTransitionAllowlist() {}
}
