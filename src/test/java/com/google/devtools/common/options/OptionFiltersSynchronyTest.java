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
package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.devtools.common.options.proto.OptionFilters;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * This test makes sure that the two java filtering enums, OptionMetadataTag and OptionEffectTag,
 * are kept in sync with the matching proto.
 */
@RunWith(JUnit4.class)
public class OptionFiltersSynchronyTest {

  @Test
  public void optionEffectTags() {
    // Check that the number of tags are equal. The proto version automatically defines an
    // UNRECOGNIZED value at -1, the sizes should actually be offset by one.
    assertThat(OptionFilters.OptionEffectTag.values())
        .hasLength(OptionEffectTag.values().length + 1);

    // Now go through each and check that the names are equal.
    for (OptionEffectTag javaTag : OptionEffectTag.values()) {
      OptionFilters.OptionEffectTag protoTag =
          OptionFilters.OptionEffectTag.forNumber(javaTag.getValue());

      // First check that the tag exists with this value, then that the names are equal.
      assertWithMessage(
              "OptionEffectTag "
                  + javaTag
                  + " does not have a proto equivalent with the same value")
          .that(protoTag)
          .isNotNull();
      assertWithMessage(
              "OptionEffectTag "
                  + javaTag
                  + " does not have the same name as the proto equivalent "
                  + protoTag)
          .that(javaTag.name())
          .isEqualTo(protoTag.name());
    }
  }

  @Test
  public void optionMetadataTags() {
    // Check that the number of tags are equal. The proto version automatically defines an
    // UNRECOGNIZED value at -1, the sizes should actually be offset by one.
    assertThat(OptionFilters.OptionMetadataTag.values())
        .hasLength(OptionMetadataTag.values().length + 1);

    // Now go through each and check that the names are equal.
    for (OptionMetadataTag javaTag : OptionMetadataTag.values()) {
      OptionFilters.OptionMetadataTag protoTag =
          OptionFilters.OptionMetadataTag.forNumber(javaTag.getValue());

      // First check that the tag exists with this value, then that the names are equal.
      assertWithMessage(
              "OptionMetadataTag "
                  + javaTag
                  + " does not have a proto equivalent with the same value")
          .that(protoTag)
          .isNotNull();
      assertWithMessage(
              "OptionMetadataTag "
                  + javaTag
                  + " does not have the same name as the proto equivalent "
                  + protoTag)
          .that(javaTag.name())
          .isEqualTo(protoTag.name());
    }
  }
}
