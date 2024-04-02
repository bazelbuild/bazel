// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.core;

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;

/**
 * Non-recursive aspect that promotes {@link OutputGroupInfo#VALIDATION} outputs to {@link
 * OutputGroupInfo#VALIDATION_TOP_LEVEL}. By requesting the latter but not the former output group,
 * validations avoid blocking test execution. (Using {@link OutputGroupInfo#DEFAULT} would
 * accomplish that as well but would be overrideable with {@code --output_groups} flag.)
 *
 * <p>Name is chosen to make for semi-sensible "ValidateTarget" aspect events.
 */
class ValidateTarget extends NativeAspectClass implements ConfiguredAspectFactory {

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return AspectDefinition.builder(this)
        .applyToFiles(true) // to grab validation outputs from file targets
        .build();
  }

  @Override
  public ConfiguredAspect create(
      Label targetLabel,
      ConfiguredTarget ct,
      RuleContext context,
      AspectParameters parameters,
      RepositoryName toolsRepository)
      throws ActionConflictException, InterruptedException {
    OutputGroupInfo outputGroupInfo = OutputGroupInfo.get(ct);
    if (outputGroupInfo != null) {
      NestedSet<Artifact> validations = outputGroupInfo.getOutputGroup(OutputGroupInfo.VALIDATION);
      if (!validations.isEmpty()) {
        return ConfiguredAspect.builder(context)
            .addOutputGroup(OutputGroupInfo.VALIDATION_TOP_LEVEL, validations)
            .build();
      }
    }
    return ConfiguredAspect.forNonapplicableTarget();
  }
}
