// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import com.google.auto.value.AutoValue;

/** A {@link TargetPattern} with a potential minus sign in front of it, signifying exclusion. */
@AutoValue
public abstract class SignedTargetPattern {
  public abstract TargetPattern pattern();

  public abstract Sign sign();

  public static SignedTargetPattern create(TargetPattern pattern, Sign sign) {
    return new AutoValue_SignedTargetPattern(pattern, sign);
  }

  /** Whether this target pattern begins with a minus sign (NEGATIVE) or not (POSITIVE). */
  public enum Sign {
    POSITIVE,
    NEGATIVE
  }

  public static SignedTargetPattern parse(String pattern, TargetPattern.Parser parser)
      throws TargetParsingException {
    if (pattern.startsWith("-")) {
      return create(parser.parse(pattern.substring(1)), SignedTargetPattern.Sign.NEGATIVE);
    } else {
      return create(parser.parse(pattern), SignedTargetPattern.Sign.POSITIVE);
    }
  }
}
