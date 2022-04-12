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

import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;

/**
 * A thin interface exposing only the warning and error reporting functionality
 * of a rule.
 *
 * <p>When a class or a method needs only this functionality but not the whole
 * {@code RuleContext}, it can use this thin interface instead.
 *
 * <p>This interface should only be implemented by {@code RuleContext}.
 */
public interface RuleErrorConsumer {

  /**
   * Consume a non-attribute-specific warning in a rule.
   */
  void ruleWarning(String message);

  /**
   * Consume a non-attribute-specific error in a rule.
   */
  void ruleError(String message);

  /**
   * Consume an attribute-specific warning in a rule.
   */
  void attributeWarning(String attrName, String message);

  /**
   * Consume an attribute-specific error in a rule.
   */
  void attributeError(String attrName, String message);

  /**
   * Convenience function to report non-attribute-specific errors in the current rule and then throw
   * a {@link RuleErrorException}, immediately exiting the current rule, and shutting down the
   * invocation in a no-keep-going build. If multiple errors are present, invoke {@link #ruleError}
   * to collect additional error information before calling this method.
   */
  // TODO(bazel-team): Consider not throwing and instead just returning the exception, thereby
  // forcing the caller to use the throw statement instead of abstracting the control flow (which
  // can hurt readability).
  default RuleErrorException throwWithRuleError(String message) throws RuleErrorException {
    ruleError(message);
    throw new RuleErrorException(message);
  }

  /** See {@link #throwWithRuleError(String)}. */
  default RuleErrorException throwWithRuleError(Throwable cause) throws RuleErrorException {
    ruleError(cause.getMessage());
    throw new RuleErrorException(cause);
  }

  /** See {@link #throwWithRuleError(String)}. */
  default RuleErrorException throwWithRuleError(String message, Throwable cause)
      throws RuleErrorException {
    ruleError(message);
    throw new RuleErrorException(message, cause);
  }

  /**
   * Convenience function to report attribute-specific errors in the current rule, and then throw a
   * {@link RuleErrorException}, immediately exiting the build invocation. Alternatively, invoke
   * {@link #attributeError} instead to collect additional error information before ending the
   * invocation.
   *
   * <p>If the name of the attribute starts with <code>$</code>
   * it is replaced with a string <code>(an implicit dependency)</code>.
   */
  default RuleErrorException throwWithAttributeError(String attrName, String message)
      throws RuleErrorException {
    attributeError(attrName, message);
    throw new RuleErrorException(message);
  }

  /**
   * Returns whether this instance is known to have errors at this point during analysis. Do not
   * call this method after the initializationHook has returned.
   */
  boolean hasErrors();

  /**
   * No-op if {@link #hasErrors} is false, throws {@link RuleErrorException} if it is true.
   * This provides a convenience to early-exit of configured target creation if there are errors.
   */
  default void assertNoErrors() throws RuleErrorException {
    if (hasErrors()) {
      throw new RuleErrorException();
    }
  }
}
