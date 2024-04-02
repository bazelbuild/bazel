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

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;

/**
 * Creates the Skyframe node of an aspect.
 */
public interface ConfiguredAspectFactory {
  /**
   * Creates the aspect based on the configured target of the associated rule.
   *
   * @param targetLabel label of the associated target
   * @param ct the ConfiguredTarget of the associated rule
   * @param context the context of the associated configured target plus all the attributes the
   *     aspect itself has defined
   * @param parameters information from attributes of the rule that have requested this
   * @param toolsRepository the name of the tools repository such as "{@literal @}bazel_tools"
   */
  ConfiguredAspect create(
      Label targetLabel,
      ConfiguredTarget ct,
      RuleContext context,
      AspectParameters parameters,
      RepositoryName toolsRepository)
      throws ActionConflictException, InterruptedException, RuleErrorException;

  /** Adds any aspect implementation-specific requirements to the given builder. */
  default void addAspectImplSpecificRequiredConfigFragments(
      RequiredConfigFragmentsProvider.Builder requiredFragments) {}
}
