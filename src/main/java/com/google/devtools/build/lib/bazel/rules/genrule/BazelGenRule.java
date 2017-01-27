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

package com.google.devtools.build.lib.bazel.rules.genrule;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.genrule.GenRuleBase;
import com.google.devtools.build.lib.syntax.Type;

/**
 * An implementation of genrule for Bazel.
 */
public class BazelGenRule extends GenRuleBase {

  @Override
  protected boolean isStampingEnabled(RuleContext ruleContext) {
    if (!ruleContext.attributes().has("stamp", Type.BOOLEAN)) {
      return false;
    }
    return ruleContext.attributes().get("stamp", Type.BOOLEAN);
  }
}
