// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/** Posted when there is a configuration transition. */
@AutoValue
public abstract class ConfigurationTransitionEvent
    implements Comparable<ConfigurationTransitionEvent>, Postable {
  public static ConfigurationTransitionEvent create(String parentChecksum, String childChecksum) {
    return new AutoValue_ConfigurationTransitionEvent(parentChecksum, childChecksum);
  }

  public abstract String parentChecksum();

  public abstract String childChecksum();

  @Override
  public final int compareTo(ConfigurationTransitionEvent that) {
    int result = parentChecksum().compareTo(that.parentChecksum());
    if (result != 0) {
      return result;
    }
    return childChecksum().compareTo(that.childChecksum());
  }
}
