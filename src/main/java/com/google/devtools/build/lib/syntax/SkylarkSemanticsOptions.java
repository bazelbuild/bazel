// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import com.google.devtools.common.options.UsesOnlyCoreTypes;
import java.io.Serializable;

/**
 * Contains options that affect Skylark's semantics.
 *
 * <p>These are injected into Skyframe when a new build invocation occurs. Changing these options
 * between builds will trigger a reevaluation of everything that depends on the Skylark
 * interpreter &mdash; in particular, processing BUILD and .bzl files.
 *
 * <p>Because these options are stored in Skyframe, they must be immutable and serializable, and so
 * are subject to the restrictions of {@link UsesOnlyCoreTypes}: No {@link Option#allowMultiple}
 * options, and no options with types not handled by the default converters. (Technically all
 * options classes are mutable because their fields are public and non-final, but we assume no one
 * is manipulating these fields by the time parsing is complete.)
 */
@UsesOnlyCoreTypes
public class SkylarkSemanticsOptions extends OptionsBase implements Serializable {
  // Used in an integration test to confirm that flags are visible to the interpreter.
  @Option(
      name = "internal_skylark_flag_test_canary",
      defaultValue = "false",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED
  )
  public boolean skylarkFlagTestCanary;

  @Option(
    name = "incompatible_depset_constructor",
    defaultValue = "false",
    category = "incompatible changes",
    help = "If set to true, disables the deprecated `set` constructor for depsets."
  )
  public boolean incompatibleDepsetConstructor;
}
