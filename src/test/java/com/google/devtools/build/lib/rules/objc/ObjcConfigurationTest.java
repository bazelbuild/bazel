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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for Objective-C configuration. */
@RunWith(JUnit4.class)
public class ObjcConfigurationTest extends BuildViewTestCase {
  @Test
  public void disallowBothHeaderThinningAndIncludeScanning() {
    InvalidConfigurationException e =
        assertThrows(
            InvalidConfigurationException.class,
            () ->
                useConfiguration(
                    "--experimental_objc_header_thinning", "--experimental_objc_include_scanning"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Only one of header thinning (--experimental_objc_header_thinning) and include "
                + "scanning (--objc_include_scanning) can be enabled.");
  }
}
