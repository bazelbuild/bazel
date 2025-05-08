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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Attribute;
import net.starlark.java.syntax.Location;

/**
 * Base class for implementations of {@link
 * com.google.devtools.build.lib.analysis.RuleErrorConsumer}.
 *
 * <p>Do not create new implementations of this class - instead, use {@link RuleContext} in Native
 * rule definitions, and {@link StarlarkErrorReporter} in Starlark API definitions. For use in
 * testing, implement {@link RuleErrorConsumer} instead.
 */
public abstract class EventHandlingErrorReporter implements RuleErrorConsumer {
  private final String ruleClassNameForLogging;
  private final AnalysisEnvironment env;

  protected EventHandlingErrorReporter(String ruleClassNameForLogging, AnalysisEnvironment env) {
    this.ruleClassNameForLogging = ruleClassNameForLogging;
    this.env = env;
  }

  private void reportError(Location location, String message) {
    // TODO(ulfjack): Consider generating the error message from the root cause event rather than
    // the other way round.
    if (!hasErrors()) {
      // We must not report duplicate events, so we only report the first one for now.
      BuildConfigurationValue configuration = getConfiguration();
      env.getEventHandler()
          .post(AnalysisRootCauseEvent.withConfigurationValue(configuration, getLabel(), message));
    }
    env.getEventHandler().handle(Event.error(location, message));
  }

  @Override
  public void ruleError(String message) {
    reportError(getRuleLocation(), prefixRuleMessage(message));
  }

  @Override
  public void attributeError(String attrName, String message) {
    reportError(getRuleLocation(), completeAttributeMessage(attrName, message));
  }

  @Override
  public boolean hasErrors() {
    return env.hasErrors();
  }

  public void reportWarning(Location location, String message) {
    env.getEventHandler().handle(Event.warn(location, message));
  }

  @Override
  public void ruleWarning(String message) {
    env.getEventHandler().handle(Event.warn(getRuleLocation(), prefixRuleMessage(message)));
  }

  @Override
  public void attributeWarning(String attrName, String message) {
    reportWarning(getRuleLocation(), completeAttributeMessage(attrName, message));
  }

  private String prefixRuleMessage(String message) {
    return String.format("in %s rule %s: %s", ruleClassNameForLogging, getLabel(), message);
  }

  private String maskInternalAttributeNames(String name) {
    return Attribute.isImplicit(name) ? "(an implicit dependency)" : name;
  }

  /**
   * Prefixes the given message with details about the rule and appends details about the macro that
   * created this rule, if applicable.
   */
  private String completeAttributeMessage(String attrName, String message) {
    // Appends a note to the given message if the offending rule was created by a macro.

    return String.format(
        "in %s attribute of %s rule %s: %s%s",
        maskInternalAttributeNames(attrName),
        ruleClassNameForLogging,
        getLabel(),
        message,
        getMacroMessageAppendix(attrName));
  }

  /** Returns a string describing the macro that created this rule, or an empty string. */
  protected abstract String getMacroMessageAppendix(String attrName);

  protected abstract Label getLabel();

  protected abstract BuildConfigurationValue getConfiguration();

  protected abstract Location getRuleLocation();
}
