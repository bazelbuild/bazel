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

import org.objectweb.asm.commons.Remapper;

/** A {@link Remapper} with a user-controlled switch which sets whether the type mapping applies. */
public final class SwitchableTypeMapper extends Remapper {

  private final TypeMapper mapper;
  private boolean isSwitchOn;

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

  public void turnOn() {
    isSwitchOn = true;
  }

  public void turnOff() {
    isSwitchOn = false;
  }
}
