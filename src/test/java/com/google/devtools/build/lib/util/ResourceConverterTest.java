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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceConverter}. */
@RunWith(JUnit4.class)
public class ResourceConverterTest {

  private ResourceConverter resourceConverter;

  @Test
  public void convertNumber_returnsInt() throws Exception {
    resourceConverter = new ResourceConverter(() -> null);
    assertThat(resourceConverter.convert("6")).isEqualTo(6);
  }

  @Test
  public void convertNumber_greaterThanMax_throwsException() {
    resourceConverter = new ResourceConverter(() -> null, 0, 1);
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> resourceConverter.convert("2"));
    assertThat(thrown).hasMessageThat().contains("cannot be greater than 1");
  }

  @Test
  public void convertNumber_lessThanMin_throwsException() throws Exception {
    resourceConverter = new ResourceConverter(() -> null, -1, 1);
    assertThat(resourceConverter.convert("0")).isEqualTo(0);
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> resourceConverter.convert("-2"));
    assertThat(thrown).hasMessageThat().contains("must be at least -1");
  }

  @Test
  public void convertAuto_returnsSuppliedAutoValue() throws Exception {
    resourceConverter = new ResourceConverter(() -> 5);
    assertThat(resourceConverter.convert("auto")).isEqualTo(5);
  }

  @Test
  public void convertAuto_withOperator_appliesOperatorToAuto() throws Exception {
    resourceConverter = new ResourceConverter(() -> 5);
    assertThat(resourceConverter.convert("auto-1")).isEqualTo(4);
  }

  @Test
  public void convertAuto_withInvalidOperator_throwsException() {
    resourceConverter = new ResourceConverter(() -> null);
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> resourceConverter.convert("auto/2"));
    assertThat(thrown).hasMessageThat().contains("does not follow correct syntax");
  }

  @Test
  public void convertAuto_isFloat_returnsRoundedInt() throws Exception {
    resourceConverter = new ResourceConverter(() -> 5);
    assertThat(resourceConverter.convert("auto*.51")).isEqualTo(3);
  }

  @Test
  public void convertHostCpus_returnsCpuSetting() throws Exception {
    LocalHostCapacity.setLocalHostCapacity(ResourceSet.createWithRamCpu(0, 15));
    resourceConverter = new ResourceConverter(() -> 5);
    assertThat(resourceConverter.convert("HOST_CPUS")).isEqualTo(15);
  }

  @Test
  public void convertRam_returnsRamSetting() throws Exception {
    LocalHostCapacity.setLocalHostCapacity(ResourceSet.createWithRamCpu(10, 0));
    resourceConverter = new ResourceConverter(() -> 5);
    assertThat(resourceConverter.convert("HOST_RAM")).isEqualTo(10);
  }

  @Test
  public void convertFloat_throwsException() {
    resourceConverter = new ResourceConverter(() -> null);
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> resourceConverter.convert(".5"));
    assertThat(thrown).hasMessageThat().contains("not an int");
  }

  @Test
  public void convertWrongKeyword_throwsException() {
    resourceConverter = new ResourceConverter(() -> null);
    OptionsParsingException thrown =
        assertThrows(
            OptionsParsingException.class, () -> resourceConverter.convert("invalid_keyword"));
    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo(
            "Parameter 'invalid_keyword' does not follow correct syntax. "
                + "This flag takes an integer, or a keyword "
                + "(\"auto\", \"HOST_CPUS\", \"HOST_RAM\"),"
                + " optionally followed by an operation ([-|*]<float>) eg. \"auto\", "
                + "\"HOST_CPUS*.5\".");
  }

  @Test
  public void convertAlmostValidKeyword_throwsException() {
    resourceConverter = new ResourceConverter(() -> null);
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> resourceConverter.convert("aut"));
    assertThat(thrown).hasMessageThat().contains("does not follow correct syntax");
  }

  @Test
  public void buildConverter_beforeResources_usesResources() throws Exception {
    resourceConverter = new ResourceConverter(() -> null);
    LocalHostCapacity.setLocalHostCapacity(ResourceSet.createWithRamCpu(0, 15));
    assertThat(resourceConverter.convert("HOST_CPUS")).isEqualTo(15);
  }

  @Test
  public void buildConverter_withNoMin_setsMinTo1() {
    resourceConverter = new ResourceConverter(() -> null);
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> resourceConverter.convert("0"));
    assertThat(thrown).hasMessageThat().contains("must be at least 1");
  }

}
