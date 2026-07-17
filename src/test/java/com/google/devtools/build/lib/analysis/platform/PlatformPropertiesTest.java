// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link PlatformProperties}. */
@RunWith(JUnit4.class)
public class PlatformPropertiesTest {
  @Test
  public void properties_empty() throws Exception {
    PlatformProperties.Builder builder = PlatformProperties.builder();
    builder.setProperties(ImmutableMap.of());
    PlatformProperties platformProperties = builder.build();

    assertThat(platformProperties).isNotNull();
    assertThat(platformProperties.properties()).isNotNull();
    assertThat(platformProperties.properties()).isEmpty();
  }

  @Test
  public void properties_one() throws Exception {
    PlatformProperties.Builder builder = PlatformProperties.builder();
    builder.setProperties(ImmutableMap.of("elem1", "value1"));
    PlatformProperties platformProperties = builder.build();

    assertThat(platformProperties).isNotNull();
    assertThat(platformProperties.properties()).isNotNull();
    assertThat(platformProperties.properties()).containsExactly("elem1", "value1");
  }

  @Test
  public void properties_parentPlatform_keep() throws Exception {
    PlatformProperties parent =
        PlatformProperties.builder().setProperties(ImmutableMap.of("parent", "properties")).build();

    PlatformProperties.Builder builder = PlatformProperties.builder();
    builder.setParent(parent);
    PlatformProperties platformProperties = builder.build();

    assertThat(platformProperties).isNotNull();
    assertThat(platformProperties.properties()).containsExactly("parent", "properties");
  }

  @Test
  public void properties_parentPlatform_inheritance() throws Exception {
    PlatformProperties parent =
        PlatformProperties.builder()
            .setProperties(
                ImmutableMap.of("p1", "keep", "p2", "delete", "p3", "parent", "p4", "del2"))
            .build();

    PlatformProperties.Builder builder = PlatformProperties.builder();
    builder.setParent(parent);
    PlatformProperties platformProperties =
        builder.setProperties(ImmutableMap.of("p2", "", "p3", "child", "p4", "")).build();

    assertThat(platformProperties).isNotNull();
    assertThat(platformProperties.properties()).containsExactly("p1", "keep", "p3", "child");
  }
}
