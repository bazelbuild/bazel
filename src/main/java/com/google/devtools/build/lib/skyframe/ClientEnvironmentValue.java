// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * The value of an environmental variable from the client environment. These are invalidated and
 * injected by {@link SequencedSkyframeExecutor}.
 */
public final class ClientEnvironmentValue implements SkyValue {
  private final String value;

  public ClientEnvironmentValue(@Nullable String value) {
    this.value = value;
  }

  /** @return the value in the client environment or null if unset in the environment. */
  @Nullable
  public String getValue() {
    return value;
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof ClientEnvironmentValue clientEnvironmentValue
        && Objects.equals(clientEnvironmentValue.value, value);
  }

  @Override
  public int hashCode() {
    return (value != null) ? value.hashCode() : 0;
  }
}
