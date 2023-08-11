// Copyright 2010 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TriState;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for link stamping. */
@RunWith(JUnit4.class)
public final class StampTest extends BuildViewTestCase {

  /** Tests that link stamping is disabled for all tests that support it. */
  @Test
  public void testNoStampingForTests() {
    for (Map.Entry<String, RuleClass> e :
        analysisMock.createRuleClassProvider().getRuleClassMap().entrySet()) {
      String name = e.getKey();
      RuleClass ruleClass = e.getValue();
      if (TargetUtils.isTestRuleName(name) && ruleClass.hasAttr("stamp", BuildType.TRISTATE)) {
        assertThat(ruleClass.getAttributeByName("stamp").getDefaultValue(null))
            .isEqualTo(TriState.NO);
      }
    }
  }
}
