// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.Objects;

/** A marker class for configuration transitions that are defined in Starlark. */
public abstract class StarlarkTransition implements ConfigurationTransition {

  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  StarlarkTransition(StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  public void replayOn(ExtendedEventHandler eventHandler) {
    starlarkDefinedConfigTransition.getEventHandler().replayOn(eventHandler);
    starlarkDefinedConfigTransition.getEventHandler().clear();
  }

  @Override
  public boolean equals(Object object) {
    if (object == this) {
      return true;
    }
    if (object instanceof StarlarkTransition) {
      StarlarkDefinedConfigTransition starlarkDefinedConfigTransition =
          ((StarlarkTransition) object).starlarkDefinedConfigTransition;
      return Objects.equals(starlarkDefinedConfigTransition, this.starlarkDefinedConfigTransition);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(starlarkDefinedConfigTransition);
  }
}
