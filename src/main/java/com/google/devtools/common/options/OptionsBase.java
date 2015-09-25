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

package com.google.devtools.common.options;

import com.google.common.escape.CharEscaperBuilder;
import com.google.common.escape.Escaper;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Base class for all options classes.  Extend this class, adding public
 * instance fields annotated with @Option.  Then you can create instances
 * either programmatically:
 *
 * <pre>
 *   X x = Options.getDefaults(X.class);
 *   x.host = "localhost";
 *   x.port = 80;
 * </pre>
 *
 * or from an array of command-line arguments:
 *
 * <pre>
 *   OptionsParser parser = OptionsParser.newOptionsParser(X.class);
 *   parser.parse("--host", "localhost", "--port", "80");
 *   X x = parser.getOptions(X.class);
 * </pre>
 *
 * <p>Subclasses of OptionsBase <b>must</b> be constructed reflectively,
 * i.e. using not {@code new MyOptions}, but one of the two methods above
 * instead.  (Direct construction creates an empty instance, not containing
 * default values.  This leads to surprising behavior and often
 * NullPointerExceptions, etc.)
 */
public abstract class OptionsBase {

  private static final Escaper ESCAPER = new CharEscaperBuilder()
      .addEscape('\\', "\\\\").addEscape('"', "\\\"").toEscaper();

  /**
   * Subclasses must provide a default (no argument) constructor.
   */
  protected OptionsBase() {
    // There used to be a sanity check here that checks the stack trace of this constructor
    // invocation; unfortunately, that makes the options construction about 10x slower. So be
    // careful with how you construct options classes.
  }

  /**
   * Returns this options object in the form of a (new) mapping from option
   * names, including inherited ones, to option values.  If the public fields
   * are mutated, this will be reflected in subsequent calls to {@code asMap}.
   * Mutation of this map by the caller does not affect this options object.
   */
  public final Map<String, Object> asMap() {
    return OptionsParserImpl.optionsAsMap(this);
  }

  @Override
  public final String toString() {
    return getClass().getName() + asMap();
  }

  /**
   * Returns a string that uniquely identifies the options. This value is
   * intended for analysis caching.
   */
  public final String cacheKey() {
    StringBuilder result = new StringBuilder(getClass().getName()).append("{");

    for (Entry<String, Object> entry : asMap().entrySet()) {
      result.append(entry.getKey()).append("=");

      Object value = entry.getValue();
      // This special case is needed because List.toString() prints the same
      // ("[]") for an empty list and for a list with a single empty string.
      if (value instanceof List<?> && ((List<?>) value).isEmpty()) {
        result.append("EMPTY");
      } else if (value == null) {
        result.append("NULL");
      } else {
        result
            .append('"')
            .append(ESCAPER.escape(value.toString()))
            .append('"');
      }
      result.append(", ");
    }

    return result.append("}").toString();
  }

  @Override
  public final boolean equals(Object that) {
    return that != null &&
        this.getClass() == that.getClass() &&
        this.asMap().equals(((OptionsBase) that).asMap());
  }

  @Override
  public final int hashCode() {
    return this.getClass().hashCode() + asMap().hashCode();
  }
}
