// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class InstrumentationOutputFactoryTest {
  @Test
  public void testInstrumentationOutputFactory_cannotCreateFactorIfLocalSupplierUnset() {
    InstrumentationOutputFactory.Builder factoryBuilder =
        new InstrumentationOutputFactory.Builder();
    factoryBuilder.setBuildEventArtifactInstrumentationOutputBuilderSupplier(
        BuildEventArtifactInstrumentationOutput.Builder::new);

    assertThrows(
        "Cannot create InstrumentationOutputFactory without localOutputBuilderSupplier",
        NullPointerException.class,
        factoryBuilder::build);
  }

  @Test
  public void testInstrumentationOutputFactory_cannotCreateFactorIfBepSupplierUnset() {
    InstrumentationOutputFactory.Builder factoryBuilder =
        new InstrumentationOutputFactory.Builder();
    factoryBuilder.setLocalInstrumentationOutputBuilderSupplier(
        LocalInstrumentationOutput.Builder::new);

    assertThrows(
        "Cannot create InstrumentationOutputFactory without bepOutputBuilderSupplier",
        NullPointerException.class,
        factoryBuilder::build);
  }

  @Test
  public void testInstrumentationOutputFactory_successfulCreateFactory() {
    InstrumentationOutputFactory.Builder factoryBuilder =
        new InstrumentationOutputFactory.Builder();
    factoryBuilder.setLocalInstrumentationOutputBuilderSupplier(
        LocalInstrumentationOutput.Builder::new);
    factoryBuilder.setBuildEventArtifactInstrumentationOutputBuilderSupplier(
        BuildEventArtifactInstrumentationOutput.Builder::new);

    InstrumentationOutputFactory factory = factoryBuilder.build();

    assertThat(factory.createLocalInstrumentationOutputBuilder()).isNotNull();
    assertThat(factory.createBuildEventArtifactInstrumentationOutputBuilder()).isNotNull();
  }
}
