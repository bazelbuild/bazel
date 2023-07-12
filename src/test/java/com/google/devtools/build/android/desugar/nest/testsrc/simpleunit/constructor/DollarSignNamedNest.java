/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.constructor;

/** The entry-point point for testing to desugar classes with '$' in class names. */
public class DollarSignNamedNest {
  public static long execute(long initialValue) {
    return $Dollar$Sign$Named$Nest$.execute(initialValue);
  }
}

class $Dollar$Sign$Named$Nest$ {

  private final long value;

  static long execute(long value) {
    return new $Dollar$Sign$Named$Nest$(value).toMember().toHost().value;
  }

  private $Dollar$Sign$Named$Nest$(long value) {
    this.value = value + 1;
  }

  $Dollar$Sign$Named$Member$ toMember() {
    return new $Dollar$Sign$Named$Member$(this);
  }

  static class $Dollar$Sign$Named$Member$ {

    private final long value;

    private $Dollar$Sign$Named$Member$($Dollar$Sign$Named$Nest$ host) {
      value = host.value + 2;
    }

    $Dollar$Sign$Named$Nest$ toHost() {
      return new $Dollar$Sign$Named$Nest$(value);
    }
  }
}
