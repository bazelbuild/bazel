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

import com.google.common.collect.ImmutableMap;
import java.util.ArrayList;
import java.util.Collections;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that we have descriptions for every option tag. */
@RunWith(JUnit4.class)
public class OptionFilterDescriptionsTest {

  @Test
  public void documentationOrderIncludesAllDocumentedCategories() {
    // Expect the documentation order to include everything but the undocumented category.
    ArrayList<OptionDocumentationCategory> docOrderPlusUndocumented = new ArrayList<>();
    Collections.addAll(docOrderPlusUndocumented, OptionFilterDescriptions.documentationOrder);
    docOrderPlusUndocumented.add(OptionDocumentationCategory.UNDOCUMENTED);

    assertThat(OptionDocumentationCategory.values())
        .asList()
        .containsExactlyElementsIn(docOrderPlusUndocumented);
  }

  @Test
  public void optionDocumentationCategoryDescriptionsContainsAllCategories() {
    // Check that we have a description for all valid option categories.
    ImmutableMap<OptionDocumentationCategory, String> optionCategoryDescriptions =
        OptionFilterDescriptions.getOptionCategoriesEnumDescription("blaze");

    assertThat(OptionDocumentationCategory.values())
        .asList()
        .containsExactlyElementsIn(optionCategoryDescriptions.keySet());
  }

  @Test
  public void optionEffectTagDescriptionsContainsAllTags() {
    // Check that we have a description for all valid option tags.
    ImmutableMap<OptionEffectTag, String> optionEffectTagDescription =
        OptionFilterDescriptions.getOptionEffectTagDescription("blaze");

    assertThat(OptionEffectTag.values())
        .asList()
        .containsExactlyElementsIn(optionEffectTagDescription.keySet());
  }

  @Test
  public void optionMetadataTagDescriptionsContainsAllTags() {
    // Check that we have a description for all valid option tags.
    ImmutableMap<OptionMetadataTag, String> optionMetadataTagDescription =
        OptionFilterDescriptions.getOptionMetadataTagDescription("blaze");

    assertThat(OptionMetadataTag.values())
        .asList()
        .containsExactlyElementsIn(optionMetadataTagDescription.keySet());
  }
}
