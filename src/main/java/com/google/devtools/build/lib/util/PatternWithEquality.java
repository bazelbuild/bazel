// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;

import java.util.regex.Pattern;

/**
 * Shim to work around the issue that {@link Pattern} doesn't implement equality. Treats pattern as
 * a {@link String} for equality-checking purposes. Thus, if two PatternWithEquality objects are
 * equal, their internal Pattern objects are necessarily equal, although the converse does not hold.
 */
public final class PatternWithEquality {
  public final Pattern pattern;

  public PatternWithEquality(Pattern pattern) {
    this.pattern = Preconditions.checkNotNull(pattern);
  }

  @Override
  public int hashCode() {
    return pattern.toString().hashCode();
  }

  @Override
  public boolean equals(Object other) {
    return (other instanceof PatternWithEquality)
        && pattern.toString().equals(((PatternWithEquality) other).pattern.toString());
  }
}
