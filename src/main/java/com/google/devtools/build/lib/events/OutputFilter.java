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

package com.google.devtools.build.lib.events;

import com.google.devtools.build.lib.util.StringEncoding;
import java.util.regex.Pattern;

/** An output filter for warnings. */
public interface OutputFilter {

  /** An output filter that matches everything. */
  OutputFilter OUTPUT_EVERYTHING = tag -> true;

  /** An output filter that matches nothing. */
  OutputFilter OUTPUT_NOTHING = tag -> false;

  /** Returns true iff the given tag matches the output filter. */
  boolean showOutput(String tag);

  /** An output filter using regular expression matching. */
  final class RegexOutputFilter implements OutputFilter {
    /** Returns an output filter for the given regex (by compiling it). */
    public static OutputFilter forRegex(String regex) {
      return new RegexOutputFilter(Pattern.compile(regex));
    }

    /** Returns an output filter for the given pattern. */
    public static OutputFilter forPattern(Pattern pattern) {
      return new RegexOutputFilter(pattern);
    }

    private final Pattern pattern;

    private RegexOutputFilter(Pattern pattern) {
      this.pattern = pattern;
    }

    @Override
    public boolean showOutput(String tag) {
      return pattern.matcher(StringEncoding.internalToUnicode(tag)).find();
    }

    @Override
    public String toString() {
      return pattern.toString();
    }
  }
}
