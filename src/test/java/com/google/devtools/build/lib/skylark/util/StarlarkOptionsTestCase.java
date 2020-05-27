// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylark.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.runtime.ClientOptions;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.StarlarkOptionsParser;
import com.google.devtools.build.lib.runtime.UiOptions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;

/** Helper base class for testing the use of Starlark-style flags. */
public class StarlarkOptionsTestCase extends BuildViewTestCase {

  private static final List<Class<? extends OptionsBase>> requiredOptionsClasses =
      ImmutableList.of(
          PackageOptions.class,
          StarlarkSemanticsOptions.class,
          KeepGoingOption.class,
          LoadingOptions.class,
          ClientOptions.class,
          UiOptions.class,
          CommonCommandOptions.class);
  private StarlarkOptionsParser starlarkOptionsParser;

  @Before
  public void setUp() throws Exception {
    optionsParser =
        OptionsParser.builder()
            .optionsClasses(
                Iterables.concat(
                    requiredOptionsClasses, ruleClassProvider.getConfigurationOptions()))
            .build();
    starlarkOptionsParser =
        StarlarkOptionsParser.newStarlarkOptionsParserForTesting(
            skyframeExecutor, reporter, PathFragment.EMPTY_FRAGMENT, optionsParser);
  }

  protected OptionsParsingResult parseStarlarkOptions(String options) throws Exception {
    starlarkOptionsParser.setResidueForTesting(Arrays.asList(options.split(" ")));
    starlarkOptionsParser.parse(new StoredEventHandler());
    return starlarkOptionsParser.getNativeOptionsParserFortesting();
  }

  private void writeBuildSetting(String type, String defaultValue, boolean isFlag)
      throws Exception {
    String flag = isFlag ? "True" : "False";

    scratch.file(
        "test/build_setting.bzl",
        "def _build_setting_impl(ctx):",
        "  return []",
        type + "_setting = rule(",
        "  implementation = _build_setting_impl,",
        "  build_setting = config." + type + "(flag=" + flag + ")",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:build_setting.bzl', '" + type + "_setting')",
        type
            + "_setting(name = 'my_"
            + type
            + "_setting', build_setting_default = "
            + defaultValue
            + ")");
  }

  protected void writeBasicIntFlag() throws Exception {
    writeBuildSetting("int", "42", true);
  }

  protected void writeBasicBoolFlag() throws Exception {
    writeBuildSetting("bool", "True", true);
  }
}
