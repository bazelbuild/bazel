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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@link RuleCodec} throws. */
@RunWith(JUnit4.class)
public class ThrowingRuleCodecTest extends BuildViewTestCase {

  @Test
  public void testCodec() throws Exception {
    scratch.file("cc/BUILD", "cc_library(name='lib', srcs = ['a.cc'])");
    Rule rule = (Rule) getTarget("//cc:lib");

    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get().getBuilder().setAllowDefaultCodec(true).build(),
            ImmutableClassToInstanceMap.of());
    try {
      objectCodecs.serialize(rule);
      throw new AssertionError("Should have thrown");
    } catch (SerializationException e) {
      assertThat(e).hasMessageThat()
          .isEqualTo(String.format(RuleCodec.SERIALIZATION_ERROR_TEMPLATE, rule));
    }
  }
}
