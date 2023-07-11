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
package com.google.devtools.build.lib.util;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;

/**
 * Static singleton holder for the string interning pool.  Doesn't use {@link String#intern}
 * because that consumes permgen space.
 */
public final class StringCanonicalizer {

  private static final Interner<String> interner = BlazeInterners.newWeakInterner();

  private StringCanonicalizer() {
  }

  /** Interns a String. */
  public static String intern(String arg) {
    return interner.intern(arg);
  }
}
