// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SlimProfileConfiguration}. */
@RunWith(JUnit4.class)
public final class SlimProfileConfigurationTest {

  private final SlimProfileConfiguration.SlimProfileConverter converter =
      new SlimProfileConfiguration.SlimProfileConverter();

  @Test
  public void testConverter_booleanTrue() throws Exception {
    SlimProfileConfiguration config = converter.convert("true");
    assertThat(config.isEnabled()).isTrue();
    assertThat(config.hasSizeLimit()).isFalse();
    assertThat(config.getSizeLimit()).isLessThan(0);
  }

  @Test
  public void testConverter_booleanFalse() throws Exception {
    SlimProfileConfiguration config = converter.convert("false");
    assertThat(config.isEnabled()).isFalse();
    assertThat(config.hasSizeLimit()).isFalse();
  }

  @Test
  public void testConverter_sizeLimit() throws Exception {
    SlimProfileConfiguration config = converter.convert("5M");
    assertThat(config.isEnabled()).isTrue();
    assertThat(config.hasSizeLimit()).isTrue();
    assertThat(config.getSizeLimit()).isEqualTo(5 * 1024 * 1024);
  }

  @Test
  public void testConverter_invalidInput() {
    assertThrows(OptionsParsingException.class, () -> converter.convert("invalid"));
  }
}
