// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.common.options.OptionsParser;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildOptionsView}. */
@RunWith(JUnit4.class)
public final class BuildOptionsViewTest {

  private static final ImmutableList<Class<? extends FragmentOptions>> BUILD_CONFIG_OPTIONS =
      ImmutableList.of(CoreOptions.class, CppOptions.class);

  private BuildOptions options;

  @Before
  public void constructBuildOptions() {
    options =
        BuildOptions.of(
            BUILD_CONFIG_OPTIONS,
            OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build());
  }

  @Test
  public void allowedGet() throws Exception {
    BuildOptionsView restrictedOptions =
        new BuildOptionsView(options, ImmutableSet.of(CoreOptions.class));
    assertThat(restrictedOptions.get(CoreOptions.class))
        .isSameInstanceAs(options.get(CoreOptions.class));
    ;
  }

  @Test
  public void prohibitedGet() throws Exception {
    BuildOptionsView restrictedOptions =
        new BuildOptionsView(options, ImmutableSet.of(CoreOptions.class));
    assertThrows(IllegalArgumentException.class, () -> restrictedOptions.get(CppOptions.class));
  }

  @Test
  public void allowedContains() throws Exception {
    BuildOptionsView restrictedOptions =
        new BuildOptionsView(options, ImmutableSet.of(CoreOptions.class));
    assertThat(restrictedOptions.contains(CoreOptions.class)).isTrue();
  }

  @Test
  public void prohibitedContains() throws Exception {
    BuildOptionsView restrictedOptions =
        new BuildOptionsView(options, ImmutableSet.of(CoreOptions.class));
    assertThrows(
        IllegalArgumentException.class, () -> restrictedOptions.contains(CppOptions.class));
  }

  @Test
  public void cloneTest() throws Exception {
    BuildOptionsView restrictedOptions =
        new BuildOptionsView(options, ImmutableSet.of(CoreOptions.class));
    BuildOptionsView clone = restrictedOptions.clone();
    assertThat(clone).isNotSameInstanceAs(restrictedOptions);
    assertThat(restrictedOptions.underlying()).isSameInstanceAs(options);
    assertThat(clone.underlying()).isNotSameInstanceAs(options);
    assertThat(clone.underlying()).isEqualTo(options);
  }
}
