// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.util;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.DefaultsPackage;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.flags.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Helper class for testing {@link DefaultsPackage}.
 */
public class DefaultsPackageUtil {

  public static String getDefaultsPackageForOptions(Class<? extends FragmentOptions> optionsClass,
      String... options) throws OptionsParsingException {
    OptionsParser parser = OptionsParser.newOptionsParser(
        ImmutableList.<Class<? extends OptionsBase>>of(
            BuildConfiguration.Options.class, optionsClass));
    parser.parse(options);

    InvocationPolicyEnforcer enforcer =
        new InvocationPolicyEnforcer(TestConstants.TEST_INVOCATION_POLICY);
    enforcer.enforce(parser);

    return DefaultsPackage.getDefaultsPackageContent(BuildOptions.of(
        ImmutableList.<Class<? extends FragmentOptions>>of(
            BuildConfiguration.Options.class, optionsClass), parser));
  }
}
