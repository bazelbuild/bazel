// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Bazel application data for the Starlark thread that performs analysis of rules and aspects. */
public class BazelRuleAnalysisThreadContext extends BazelStarlarkContext {

  private final RuleContext ruleContext;

  /**
   * Constructs a {@link BazelRuleAnalysisThreadContext}.
   *
   * @param ruleContext is the {@link RuleContext} of the rule for analysis of a rule or aspect
   */
  public BazelRuleAnalysisThreadContext(RuleContext ruleContext) {
    super(Phase.ANALYSIS);
    this.ruleContext = ruleContext;
  }

  /** Returns the label of the rule. */
  @Nullable
  public Label getAnalysisRuleLabel() {
    return ruleContext.getLabel();
  }

  @Override
  public String getContextForUncheckedException() {
    return ruleContext.getLabel().toString();
  }

  public RuleContext getRuleContext() {
    return ruleContext;
  }

  /**
   * Retrieves this context from a Starlark thread.
   *
   * @param thread the {@link StarlarkThread} from which to retrieve the context
   * @param what information to include in the error thrown
   * @throws EvalException if not found
   */
  @CanIgnoreReturnValue
  public static BazelRuleAnalysisThreadContext fromOrFail(StarlarkThread thread, String what)
      throws EvalException {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (ctx instanceof BazelRuleAnalysisThreadContext) {
      return (BazelRuleAnalysisThreadContext) ctx;
    }
    throw Starlark.errorf("%s can only be called from a rule or aspect implementation", what);
  }
}
