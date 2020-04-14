/*
 *
 *  Copyright 2020 The Bazel Authors. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import org.objectweb.asm.commons.Remapper;

/**
 * A {@link Remapper} with a user-controlled switch which sets whether the type mapping applies.
 *
 * <p>The underlying switch is non-reentrant and a user is expected to match its preconfigured
 * reason before changing the switch state.
 */
public final class SwitchableTypeMapper<R> extends Remapper {

  private final TypeMapper mapper;
  private boolean isSwitchOn;
  private R activeReason;

  public SwitchableTypeMapper(TypeMapper mapper) {
    this.mapper = mapper;
  }

  @Override
  public String map(String binaryName) {
    return map(ClassName.create(binaryName)).binaryName();
  }

  public ClassName map(ClassName className) {
    return isSwitchOn() ? className.acceptTypeMapper(mapper) : className;
  }

  public boolean isSwitchOn() {
    return isSwitchOn;
  }

  public void turnOn(R reason) {
    checkState(
        activeReason == null,
        "Expected the switch is off without any pre-existing reason, but the switch is on with"
            + " existing reason (%s), new reason(%s)",
        activeReason,
        reason);
    activeReason = checkNotNull(reason);
    isSwitchOn = true;
  }

  public void turnOff(R reason) {
    checkNotNull(
        activeReason,
        "Expected the switch is on with a matched reason, but the switch is off, new"
            + " reason(%s)",
        reason);
    checkState(
        activeReason.equals(reason),
        "Expected the switch is on with a matched reason, but the existing reason (%s) mismatches"
            + " new reason (%s)",
        activeReason,
        reason);
    activeReason = null;
    isSwitchOn = false;
  }
}
