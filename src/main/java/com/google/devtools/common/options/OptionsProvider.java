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

import com.google.devtools.common.options.OptionsParser.OptionValueDescription;
import com.google.devtools.common.options.OptionsParser.UnparsedOptionValueDescription;

import java.util.List;

/**
 * A read-only interface for options parser results, which does not allow any
 * further parsing of options.
 */
public interface OptionsProvider extends OptionsClassProvider {

  /**
   * Returns an immutable copy of the residue, that is, the arguments that
   * have not been parsed.
   */
  List<String> getResidue();

  /**
   * Returns if the named option was specified explicitly in a call to parse.
   */
  boolean containsExplicitOption(String string);

  /**
   * Returns a mutable copy of the list of all options that were specified
   * either explicitly or implicitly. These options are sorted by priority, and
   * by the order in which they were specified. If an option was specified
   * multiple times, it is included in the result multiple times. Does not
   * include the residue.
   *
   * <p>The returned list can be filtered if undocumented, hidden or implicit
   * options should not be displayed.
   */
  List<UnparsedOptionValueDescription> asListOfUnparsedOptions();

  /**
   * Returns a list of all explicitly specified options, suitable for logging
   * or for displaying back to the user. These options are sorted by priority,
   * and by the order in which they were specified. If an option was
   * explicitly specified multiple times, it is included in the result
   * multiple times. Does not include the residue.
   *
   * <p>The list includes undocumented options.
   */
  List<UnparsedOptionValueDescription> asListOfExplicitOptions();

  /**
   * Returns a list of all options, including undocumented ones, and their
   * effective values. There is no guaranteed ordering for the result.
   */
  List<OptionValueDescription> asListOfEffectiveOptions();

  /**
   * Canonicalizes the list of options that this OptionsParser has parsed. The
   * contract is that if the returned set of options is passed to an options
   * parser with the same options classes, then that will have the same effect
   * as using the original args (which are passed in here), except for cosmetic
   * differences.
   */
  List<String> canonicalize();
}
