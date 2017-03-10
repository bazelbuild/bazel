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

package com.google.devtools.build.lib.rules.platform;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;
import java.util.Map;

/** Defines a platform for execution contexts. */
public class Platform implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    Iterable<ConstraintValueProvider> constraintValues =
        ruleContext.getPrerequisites(
            PlatformRule.CONSTRAINT_VALUES_ATTR, Mode.DONT_CHECK, ConstraintValueProvider.class);

    // Verify the constraints - no two values for the same setting, and construct the map.
    ImmutableMap<ConstraintSettingProvider, ConstraintValueProvider> constraints =
        validateConstraints(ruleContext, constraintValues);
    if (constraints == null) {
      // An error occurred, return null.
      return null;
    }

    PlatformProvider.Builder platformProviderBuilder = PlatformProvider.builder();
    platformProviderBuilder.constraints(constraints);

    Map<String, String> remoteExecutionProperties =
        ruleContext.attributes().get(PlatformRule.REMOTE_EXECUTION_PROPS_ATTR, Type.STRING_DICT);
    platformProviderBuilder.remoteExecutionProperties(remoteExecutionProperties);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addProvider(PlatformProvider.class, platformProviderBuilder.build())
        .build();
  }

  private ImmutableMap<ConstraintSettingProvider, ConstraintValueProvider> validateConstraints(
      RuleContext ruleContext, Iterable<ConstraintValueProvider> constraintValues) {
    Multimap<ConstraintSettingProvider, ConstraintValueProvider> constraints =
        ArrayListMultimap.create();

    for (ConstraintValueProvider constraintValue : constraintValues) {
      constraints.put(constraintValue.constraint(), constraintValue);
    }

    // Are there any settings with more than one value?
    boolean foundError = false;
    for (ConstraintSettingProvider constraintSetting : constraints.keySet()) {
      if (constraints.get(constraintSetting).size() > 1) {
        foundError = true;
        // error
        StringBuilder constraintValuesDescription = new StringBuilder();
        for (ConstraintValueProvider constraintValue : constraints.get(constraintSetting)) {
          if (constraintValuesDescription.length() > 0) {
            constraintValuesDescription.append(", ");
          }
          constraintValuesDescription.append(constraintValue.value());
        }
        ruleContext.attributeError(
            PlatformRule.CONSTRAINT_VALUES_ATTR,
            String.format(
                "Duplicate constraint_values for constraint_setting %s: %s",
                constraintSetting.constraintSetting(), constraintValuesDescription.toString()));
      }
    }

    if (foundError) {
      return null;
    }

    // Convert to a flat map.
    ImmutableMap.Builder<ConstraintSettingProvider, ConstraintValueProvider> builder =
        new ImmutableMap.Builder<>();
    for (ConstraintSettingProvider constraintSetting : constraints.keySet()) {
      ConstraintValueProvider constraintValue =
          Iterables.getOnlyElement(constraints.get(constraintSetting));
      builder.put(constraintSetting, constraintValue);
    }

    return builder.build();
  }
}
