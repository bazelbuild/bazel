// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.packages.AttributeTransitionData;

/**
 * CcToolchainInputsTransition allows file inputs to the cc_toolchain to be transitioned
 * either between the exec or target platform (i.e. no transition).
 */
public class CcToolchainInputsTransitionFactory implements TransitionFactory<AttributeTransitionData> {

  @Override
  public ConfigurationTransition create(AttributeTransitionData data) {
    if (!data.attributes().has("target_transition_for_inputs")) {
      return ExecutionTransitionFactory.create().create(data);
    }
    boolean targetTransitionForInputs =
        data.attributes().get("target_transition_for_inputs", BOOLEAN);
    if (targetTransitionForInputs) {
      return NoTransition.INSTANCE;
    } else {
      return ExecutionTransitionFactory.create().create(data);
    }
  }

  @Override
  public TransitionType transitionType() {
    return TransitionType.ATTRIBUTE;
  }

}
