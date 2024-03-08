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

import java.util.List;
import java.util.Map;

/**
 * A read-only interface for options parser results, which does not allow any
 * further parsing of options.
 */
public interface OptionsParsingResult extends OptionsProvider {

  /**
   * Returns a description of the option value.
   *
   * @return value for the option or null if the value has not been set
   * @throws IllegalArgumentException if there is no option by the given name
   */
  OptionValueDescription getOptionValueDescription(String name);

  /**
   * Returns an immutable copy of all arguments that were skipped because they matched a skipped
   * prefix.
   */
  List<String> getSkippedArgs();

  /** Returns an immutable copy of the residue, that is, the arguments that have not been parsed. */
  List<String> getResidue();

  /**
   * Returns an immutable copy of the residue before the " -- " signals the remainder of the
   * arguments are not options.
   */
  List<String> getPreDoubleDashResidue();

  /**
   * Returns if the named option was specified explicitly in a call to parse.
   */
  boolean containsExplicitOption(String string);

  /**
   * Returns a mutable copy of the list of all options that were specified either explicitly or
   * implicitly. These options are sorted by priority, and by the order in which they were
   * specified. If an option was specified multiple times, it is included in the result multiple
   * times. Does not include the residue.
   *
   * <p>The returned list includes undocumented, hidden or implicit options, and should be filtered
   * as needed. Since it includes all options parsed, it will also include both an expansion option
   * and the options it expanded to, and so blindly using this list for a new invocation will cause
   * double-application of these options.
   */
  // TODO(b/150222792): make this aware of starlark options.
  List<ParsedOptionDescription> asCompleteListOfParsedOptions();

  /**
   * Returns a list of all explicitly specified options, suitable for logging or for displaying back
   * to the user. These options are sorted by priority, and by the order in which they were
   * specified. If an option was explicitly specified multiple times, it is included in the result
   * multiple times. Does not include the residue.
   *
   * <p>The list includes undocumented options.
   */
  // TODO(b/150222792): make this aware of Starlark options. This might be tricky because we don't
  // store Starlark option values that are explicitly specified to the same value as the default.
  List<ParsedOptionDescription> asListOfExplicitOptions();

  /**
   * Returns a list of the parsed options whose values are in the final value of the option, i.e.
   * the options that were added explicitly, expanded if necessary to the valued options they
   * affect. This will not include values that were set and then overridden by a later value of the
   * same option.
   *
   * <p>The list includes undocumented options.
   */
  // TODO(b/150222792): make this aware of Starlark options.
  List<ParsedOptionDescription> asListOfCanonicalOptions();

  /**
   * Returns a list of all options, including undocumented ones, and their effective values. There
   * is no guaranteed ordering for the result.
   */
  // TODO(b/150222792): make this aware of Starlark options
  List<OptionValueDescription> asListOfOptionValues();

  /**
   * Canonicalizes the list of options that this OptionsParser has parsed.
   *
   * <p>The contract is that if the returned set of options is passed to an options parser with the
   * same options classes, then that will have the same effect as using the original args (which are
   * passed in here), except for cosmetic differences. We do not guarantee that the 'canonical' list
   * is unique, since some flags may have effects unknown to the parser (--config, for Bazel), so we
   * do not reorder flags to further simplify the list.
   */
  List<String> canonicalize();

  /**
   * Returns a map of all the flag aliases that have been registered using --flag_alias. The keys
   * are the aliases and the values are the actual flag names.
   */
  Map<String, String> getAliases();

  /** Returns true if there were no errors during options parsing. */
  boolean success();
}
