// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link JavaConfiguration}.
 */
@RunWith(JUnit4.class)
public class JavaConfigurationTest extends ConfigurationTestCase {

  @Test
  public void testExperimentalBytecodeOptimizersFlag() throws Exception {
    InvalidConfigurationException thrown =
        assertThrows(
            InvalidConfigurationException.class,
            () -> create("--experimental_bytecode_optimizers=somekey,somevalue"));
    assertThat(thrown).hasMessageThat().contains("can only accept exactly one mapping");
  }
}
