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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;

/**
 * Implementation for the "ios_framework_binary" rule.
 */
public class IosFrameworkBinary extends BinaryLinkingTargetFactory {
  public IosFrameworkBinary() {
    super(HasReleaseBundlingSupport.NO, XcodeProductType.LIBRARY_STATIC);
  }

  @Override
  protected ExtraLinkArgs getExtraLinkArgs(RuleContext ruleContext) {
    String frameworkName = getFrameworkName(ruleContext);

    return new ExtraLinkArgs(
        "-dynamiclib",
        String.format("-install_name @rpath/%1$s.framework/%1$s", frameworkName));
  }

  private String getFrameworkName(RuleContext ruleContext) {
    return ruleContext.getLabel().getName();
  }

  @Override
  protected void configureTarget(RuleConfiguredTargetBuilder target, RuleContext ruleContext) {
    IosFrameworkProvider frameworkProvider =
        new IosFrameworkProvider(getFrameworkName(ruleContext));

    target.addProvider(IosFrameworkProvider.class, frameworkProvider);
  }
}
