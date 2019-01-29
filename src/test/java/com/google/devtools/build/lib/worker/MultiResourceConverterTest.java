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
package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.build.lib.worker.WorkerOptions.MultiResourceConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link MultiResourceConverter}. */
@RunWith(JUnit4.class)
public class MultiResourceConverterTest {

  public MultiResourceConverter multiResourceConverter;
  public ResourceConverter resourceConverter;

  @Before
  public void setUp() {
    multiResourceConverter = new MultiResourceConverter();
    resourceConverter = new ResourceConverter(() -> null);
  }

  @Test
  public void convert_mnemonicEqualsAuto_returnsDefault() throws OptionsParsingException {
    assertThat(multiResourceConverter.convert("someMnemonic=auto").getValue())
        .isEqualTo(MultiResourceConverter.DEFAULT_VALUE);
  }

  @Test
  public void convert_mnemonicEqualsKeyword_equalsResourceConverterConvertKeyword()
      throws OptionsParsingException {
    assertThat(multiResourceConverter.convert("someMnemonic=HOST_CPUS-1").getValue())
        .isEqualTo(resourceConverter.convert("HOST_CPUS-1"));
  }

  @Test
  public void convert_auto_returnsDefault() throws OptionsParsingException {
    assertThat(multiResourceConverter.convert("auto").getValue())
        .isEqualTo(MultiResourceConverter.DEFAULT_VALUE);
  }

  @Test
  public void convert_keyword_equalsResourceConverterConvertKeyword()
      throws OptionsParsingException {
    assertThat(multiResourceConverter.convert("HOST_CPUS-1").getValue())
        .isEqualTo(resourceConverter.convert("HOST_CPUS-1"));
  }

  @Test
  public void convert_mnemonic_savesCorrectKey() throws Exception {
    assertThat(multiResourceConverter.convert("someMnemonic=10").getKey())
        .isEqualTo("someMnemonic");
  }

  @Test
  public void convert_auto_setsEmptyStringAKADefaultAsKey() throws Exception {
    assertThat(multiResourceConverter.convert("auto").getKey()).isEmpty();
  }
}
