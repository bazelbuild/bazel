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

package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.skylarkinterface.ParamType;

/** A value class to store {@link ParamType} metadata to avoid using Java proxies. */
public final class ParamTypeDescriptor {

  private final Class<?> type;
  private final Class<?> generic1;

  private ParamTypeDescriptor(Class<?> type, Class<?> generic1) {
    this.type = type;
    this.generic1 = generic1;
  }

  /** @see ParamType#type() */
  public Class<?> getType() {
    return type;
  }

  /** @see ParamType#generic1() */
  public Class<?> getGeneric1() {
    return generic1;
  }

  static ParamTypeDescriptor of(ParamType paramType) {
    return new ParamTypeDescriptor(paramType.type(), paramType.generic1());
  }
}
