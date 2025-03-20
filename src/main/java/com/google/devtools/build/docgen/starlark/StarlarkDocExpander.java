// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.docgen.starlark;

import com.google.devtools.build.docgen.RuleLinkExpander;

/** A utility class for replacing variables in documentation strings with their actual values. */
public class StarlarkDocExpander {

  public final RuleLinkExpander ruleExpander;

  public StarlarkDocExpander(RuleLinkExpander ruleExpander) {
    this.ruleExpander = ruleExpander;
  }

  public String expand(String docString) {
    return ruleExpander.expand(
        StarlarkDocUtils.substituteVariables(docString, ruleExpander.beRoot()));
  }
}
