// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SdkFrameworksDependerRule;

/**
 * Common logic for rules that inherit from {@link SdkFrameworksDependerRule}.
 */
public class ObjcSdkFrameworks {

  /**
   * Class that handles extraction and processing of attributes common to inheritors of {@link
   * SdkFrameworksDependerRule}.
   */
  public static class Attributes {

    private final RuleContext ruleContext;

    public Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Returns the SDK frameworks defined on the rule's {@code sdk_frameworks} attribute as well as
     * base frameworks defined in {@link ObjcRuleClasses#AUTOMATIC_SDK_FRAMEWORKS}.
     */
    ImmutableSet<SdkFramework> sdkFrameworks() {
      ImmutableSet.Builder<SdkFramework> result = new ImmutableSet.Builder<>();
      result.addAll(ObjcRuleClasses.AUTOMATIC_SDK_FRAMEWORKS);
      for (String explicit : ruleContext.attributes().get("sdk_frameworks", Type.STRING_LIST)) {
        result.add(new SdkFramework(explicit));
      }
      return result.build();
    }

    /**
     * Returns all SDK frameworks defined on the rule's {@code weak_sdk_frameworks} attribute.
     */
    ImmutableSet<SdkFramework> weakSdkFrameworks() {
      ImmutableSet.Builder<SdkFramework> result = new ImmutableSet.Builder<>();
      for (String frameworkName :
          ruleContext.attributes().get("weak_sdk_frameworks", Type.STRING_LIST)) {
        result.add(new SdkFramework(frameworkName));
      }
      return result.build();
    }

    /**
     * Returns all SDK dylibs defined on the rule's {@code sdk_dylibs} attribute.
     */
    ImmutableSet<String> sdkDylibs() {
      return ImmutableSet.copyOf(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST));
    }
  }
}
