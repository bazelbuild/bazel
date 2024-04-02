// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen.testutil;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Rule definitions that can be used for testing.
 */
public class TestData {
  public static class TestRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder().name("testrule")
          .factoryClass(DummyRuleFactory.class).ancestors(IntermediateRule.class).build();
    }
  }

  public static class IntermediateRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder.build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder().name("testrule")
          .factoryClass(DummyRuleFactory.class).ancestors(BaseRule.class).build();
    }
  }

  public static class BaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder.build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder().name("base_rule")
          .factoryClass(DummyRuleFactory.class).build();
    }
  }

  public static class DummyRuleFactory implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      throw new IllegalStateException();
    }
  }
}

