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
package com.google.devtools.build.lib.analysis.util;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Default behaviors for {@link MockRule}.
 *
 * <p>All of these can be optionally added or overridden for specific mock rules.
 */
public class MockRuleDefaults {
  /**
   * Stock <code>"deps"</code> attribute for rule classes that don't need special behavior.
   */
  public static final Attribute.Builder<?> DEPS_ATTRIBUTE =
      attr("deps", BuildType.LABEL_LIST).allowedFileTypes();

  /**
   * The default attributes added to all mock rules.
   *
   * <p>Does not apply when {@link MockRule#ancestor} is set.
   */
  public static final ImmutableList<Attribute.Builder<?>> DEFAULT_ATTRIBUTES = ImmutableList.of(
      attr("testonly", BOOLEAN).nonconfigurable("test").value(false),
      attr("deprecation", STRING).nonconfigurable("test").value((String) null),
      attr("tags", STRING_LIST),
      attr("visibility", NODEP_LABEL_LIST).orderIndependent().cfg(HOST).nonconfigurable("test"),
      attr(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST)
          .allowedFileTypes(FileTypeSet.NO_FILE)
          .dontCheckConstraints(),
      attr(RuleClass.RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST)
          .allowedFileTypes(FileTypeSet.NO_FILE)
          .dontCheckConstraints());

  /**
   * The default configured target factory for mock rules.
   *
   * <p>Can be overridden with {@link MockRule#factory}.
   * */
  public static class DefaultConfiguredTargetFactory implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
      NestedSet<Artifact> filesToBuild =
          NestedSetBuilder.wrap(Order.STABLE_ORDER, ruleContext.getOutputArtifacts());
      for (Artifact artifact : ruleContext.getOutputArtifacts()) {
        ruleContext.registerAction(
            FileWriteAction.createEmptyWithInputs(
                ruleContext.getActionOwner(), ImmutableList.of(), artifact));
      }
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(filesToBuild)
          .setRunfilesSupport(null, null)
          .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
          .build();
    }
  }
}
