// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for experimental_enable_tools_defaults_package flag. TODO(dbabkin): remove after
 * //tools/defaults package gone.
 */
@RunWith(JUnit4.class)
public class EnableDefaultsPackageOptionTest extends BuildViewTestCase {

  @Test
  public void testEnableDefaultsPackageOptionWorks() throws Exception {
    // do not need that as value is true by default.
    // setPackageCacheOptions("--experimental_enable_tools_defaults_package=true");

    ConfiguredTarget target = getConfiguredTarget("//tools/defaults:jdk");

    assertThat(target.getLabel().toString()).isEqualTo("//tools/defaults:jdk");
  }

  @Test
  public void testDisabledDefaultsPackageOptionWorks() throws Exception {

    scratch.file(
        "a/BUILD",
        "filegroup(",
        "  name = 'my_filegroup',",
        "  srcs = ['//tools/defaults:jdk'],",
        ")");

    reporter.removeHandler(failFastHandler);
    setPackageCacheOptions("--experimental_enable_tools_defaults_package=false");
    ConfiguredTarget target = getConfiguredTarget("//a:my_filegroup");

    assertThat(target).isNull();
    assertContainsEvent(
        "no such package 'tools/defaults': "
            + "BUILD file not found on package path and referenced by '//a:my_filegroup'");
  }

  @Test
  public void testFlipFlagOnFly() throws Exception {

    setPackageCacheOptions("--experimental_enable_tools_defaults_package=false");
    ConfiguredTarget defaultsJDKtarget = getConfiguredTarget("//tools/defaults:jdk");
    assertThat(defaultsJDKtarget).isNull();

    setPackageCacheOptions("--experimental_enable_tools_defaults_package=true");
    defaultsJDKtarget = getConfiguredTarget("//tools/defaults:jdk");
    assertThat(defaultsJDKtarget).isNotNull();
    assertThat(defaultsJDKtarget.getLabel().toString()).isEqualTo("//tools/defaults:jdk");

    setPackageCacheOptions("--experimental_enable_tools_defaults_package=false");
    defaultsJDKtarget = getConfiguredTarget("//tools/defaults:jdk");
    assertThat(defaultsJDKtarget).isNull();
  }
}
