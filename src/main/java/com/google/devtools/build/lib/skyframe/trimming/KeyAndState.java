// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.trimming;

import com.google.auto.value.AutoValue;

/**
 * A pair of some key type and a valid/invalid state, for use in {@link TrimmedConfigurationCache}.
 */
@AutoValue
abstract class KeyAndState<KeyT> {
  enum State {
    VALID(true),
    POSSIBLY_INVALID(false);

    private final boolean isKnownValid;

    State(boolean isKnownValid) {
      this.isKnownValid = isKnownValid;
    }

    boolean isKnownValid() {
      return isKnownValid;
    }
  }

  abstract KeyT getKey();

  abstract State getState();

  static <KeyT> KeyAndState<KeyT> create(KeyT key) {
    return create(key, State.VALID);
  }

  private static <KeyT> KeyAndState<KeyT> create(KeyT key, State state) {
    return new AutoValue_KeyAndState<>(key, state);
  }

  KeyAndState<KeyT> asValidated() {
    if (State.VALID.equals(getState())) {
      return this;
    }
    return create(getKey(), State.VALID);
  }

  KeyAndState<KeyT> asInvalidated() {
    if (State.POSSIBLY_INVALID.equals(getState())) {
      return this;
    }
    return create(getKey(), State.POSSIBLY_INVALID);
  }
}
