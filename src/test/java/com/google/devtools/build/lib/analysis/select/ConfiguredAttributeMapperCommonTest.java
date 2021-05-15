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
package com.google.devtools.build.lib.analysis.select;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that {@link ConfiguredAttributeMapper} fulfills all behavior expected
 * of {@link AbstractAttributeMapper}.
 *
 * <p>This is distinct from {@link
 * com.google.devtools.build.lib.analysis.ConfiguredAttributeMapperTest} because the latter needs to
 * inherit from {@link com.google.devtools.build.lib.analysis.util.BuildViewTestCase} to run tests
 * with build configurations.
 */
@RunWith(JUnit4.class)
public class ConfiguredAttributeMapperCommonTest extends AbstractAttributeMapperTest {
  @Before
  public final void createMapper() throws Exception {
    mapper = ConfiguredAttributeMapper.of(rule, ImmutableMap.of(), targetConfig.checksum());
  }
}
