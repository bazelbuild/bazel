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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Identifier for {@link NinjaScope} to allow serialization of collection of
 * {@link NinjaTarget}, referring to Ninja scopes.
 */
public final class NinjaScopeId implements Comparable<NinjaScopeId> {
  private final AtomicInteger idGenerator;
  private final int number;

  private NinjaScopeId(AtomicInteger idGenerator) {
    this.number = idGenerator.incrementAndGet();
    this.idGenerator = idGenerator;
  }

  public static NinjaScopeId createNewScope() {
    return new NinjaScopeId(new AtomicInteger(0));
  }

  public NinjaScopeId createChild() {
    return new NinjaScopeId(this.idGenerator);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    NinjaScopeId that = (NinjaScopeId) o;
    return number == that.number;
  }

  @Override
  public int hashCode() {
    return Objects.hash(number);
  }

  @Override
  public int compareTo(NinjaScopeId o) {
    return Integer.compare(this.number, o.number);
  }
}
