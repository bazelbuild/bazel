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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.EventHandlingErrorReporter;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * {@link RuleErrorConsumer} for Native implementations of Skylark APIs.
 *
 * <p>This class largely reproduces the functionality of the error consumer in {@link
 * com.google.devtools.build.lib.analysis.RuleContext}, but should be easy to create and use from
 * methods annotated with {@link com.google.devtools.build.lib.skylarkinterface.SkylarkCallable}.
 * (Environment and Location are automatically provided to those methods be specifying the {@link
 * SkylarkCallable#useLocation()} and {@link SkylarkCallable#useEnvironment()} fields in the
 * annotation.
 *
 * <p>This class is AutoClosable, to ensure that {@link RuleErrorException} are checked and handled
 * before leaving native code. The {@link #close()} method will only throw {@link EvalException},
 * properly wrapping any {@link RuleErrorException} instances if needed.
 */
public class SkylarkErrorReporter extends EventHandlingErrorReporter implements AutoCloseable {
  private final Location location;
  private final Label label;

  public static SkylarkErrorReporter from(
      ActionConstructionContext context, Location location, Environment env) {
    return new SkylarkErrorReporter(
        context.getActionOwner().getTargetKind(),
        context.getAnalysisEnvironment(),
        location,
        env.getCallerLabel());
  }

  private SkylarkErrorReporter(
      String ruleClassNameForLogging,
      AnalysisEnvironment analysisEnvironment,
      Location location,
      Label label) {
    super(ruleClassNameForLogging, analysisEnvironment);
    this.label = label;
    this.location = location;
  }

  @Override
  protected String getMacroMessageAppendix(String attrName) {
    return "";
  }

  @Override
  protected Label getLabel() {
    return label;
  }

  @Override
  protected Location getRuleLocation() {
    return location;
  }

  @Override
  protected Location getAttributeLocation(String attrName) {
    return location;
  }

  @Override
  public void close() throws EvalException {
    try {
      assertNoErrors();
    } catch (RuleErrorException e) {
      throw new EvalException(location, e);
    }
  }
}
