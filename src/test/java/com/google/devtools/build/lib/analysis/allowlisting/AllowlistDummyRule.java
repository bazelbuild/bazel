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

package com.google.devtools.build.lib.analysis.allowlisting;

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import javax.annotation.Nullable;

/** Definition of a test rule that uses allowlists. */
public final class AllowlistDummyRule {
  public static final MockRule DEFINITION =
      () ->
          MockRule.factory(RuleFactory.class)
              .define(
                  "rule_with_allowlist",
                  (builder, env) ->
                      builder.add(
                          Allowlist.getAttributeFromAllowlistName("dummy")
                              .value(Label.parseCanonicalUnchecked("//allowlist:allowlist"))));

  /** Has to be public to make factory initialization logic happy. **/
  public static class RuleFactory implements RuleConfiguredTargetFactory {
    @Override
    @Nullable
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      if (!Allowlist.isAvailable(ruleContext, "dummy")) {
        ruleContext.ruleError("Dummy is not available.");
      }
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
          .addProvider(RunfilesProvider.EMPTY)
          .build();
    }
  }

  private AllowlistDummyRule() {}
}
