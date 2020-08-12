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
// limitations under the License
package com.google.devtools.build.lib.analysis.starlark;

import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;

/**
 * {@link RuleErrorConsumer} for Native implementations of Starlark APIs.
 *
 * <p>This class proxies reported errors and warnings to a proxy {@link RuleErrorConsumer}, except
 * that it suppresses all cases of actually throwing exceptions until this reporter is closed.
 *
 * <p>This class is AutoClosable, to ensure that {@link RuleErrorException} are checked and handled
 * before leaving native code. The {@link #close()} method will only throw {@link EvalException},
 * properly wrapping any {@link RuleErrorException} instances if needed.
 */
public class StarlarkErrorReporter implements AutoCloseable, RuleErrorConsumer {
  private final RuleErrorConsumer ruleErrorConsumer;

  public static StarlarkErrorReporter from(RuleErrorConsumer ruleErrorConsumer) {
    return new StarlarkErrorReporter(ruleErrorConsumer);
  }

  private StarlarkErrorReporter(RuleErrorConsumer ruleErrorConsumer) {
    this.ruleErrorConsumer = ruleErrorConsumer;
  }

  @Override
  public void close() throws EvalException {
    try {
      assertNoErrors();
    } catch (RuleErrorException e) {
      throw Starlark.errorf("error occurred while evaluating builtin function: %s", e.getMessage());
    }
  }

  @Override
  public void ruleWarning(String message) {
    ruleErrorConsumer.ruleWarning(message);
  }

  @Override
  public void ruleError(String message) {
    ruleErrorConsumer.ruleError(message);
  }

  @Override
  public void attributeWarning(String attrName, String message) {
    ruleErrorConsumer.attributeWarning(attrName, message);
  }

  @Override
  public void attributeError(String attrName, String message) {
    ruleErrorConsumer.attributeError(attrName, message);
  }

  @Override
  public boolean hasErrors() {
    return ruleErrorConsumer.hasErrors();
  }
}
