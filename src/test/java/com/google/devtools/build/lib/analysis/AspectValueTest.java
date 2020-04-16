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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.analysis.util.TestAspects.AttributeAspect;
import com.google.devtools.build.lib.analysis.util.TestAspects.ExtraAttributeAspect;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.skyframe.AspectValueKey;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.skyframe.AspectValue}.
 */
@RunWith(JUnit4.class)
public class AspectValueTest extends AnalysisTestCase {

  @Test
  public void keyEquality() throws Exception {
    update();
    BuildConfiguration c1 = getTargetConfiguration();
    BuildConfiguration c2 = getHostConfiguration();
    Label l1 = Label.parseAbsolute("//a:l1", ImmutableMap.of());
    Label l1b = Label.parseAbsolute("//a:l1", ImmutableMap.of());
    Label l2 = Label.parseAbsolute("//a:l2", ImmutableMap.of());
    AspectParameters i1 = new AspectParameters.Builder()
        .addAttribute("foo", "bar")
        .build();
    AspectParameters i1b = new AspectParameters.Builder()
        .addAttribute("foo", "bar")
        .build();
    AspectParameters i2 = new AspectParameters.Builder()
        .addAttribute("foo", "baz")
        .build();
    AttributeAspect a1 = TestAspects.ATTRIBUTE_ASPECT;
    AttributeAspect a1b = TestAspects.ATTRIBUTE_ASPECT;
    ExtraAttributeAspect a2 = TestAspects.EXTRA_ATTRIBUTE_ASPECT;

    // label: //a:l1 or //a:l2
    // aspectConfiguration: target or host
    // baseConfiguration: target or host
    // aspect: Attribute or ExtraAttribute
    // parameters: bar or baz

    new EqualsTester()
        .addEqualityGroup(
            createKey(l1, c1, a1, i1, c1),
            createKey(l1, c1, a1, i1b, c1),
            createKey(l1, c1, a1b, i1, c1),
            createKey(l1, c1, a1b, i1b, c1),
            createKey(l1b, c1, a1, i1, c1),
            createKey(l1b, c1, a1, i1b, c1),
            createKey(l1b, c1, a1b, i1, c1),
            createKey(l1b, c1, a1b, i1b, c1))
        .addEqualityGroup(
            createKey(l1, c1, a1, i2, c1),
            createKey(l1, c1, a1b, i2, c1),
            createKey(l1b, c1, a1, i2, c1),
            createKey(l1b, c1, a1b, i2, c1))
        .addEqualityGroup(
            createKey(l1, c1, a2, i1, c1),
            createKey(l1, c1, a2, i1b, c1),
            createKey(l1b, c1, a2, i1, c1),
            createKey(l1b, c1, a2, i1b, c1))
        .addEqualityGroup(
            createKey(l1, c1, a2, i2, c1),
            createKey(l1b, c1, a2, i2, c1))
        .addEqualityGroup(
            createKey(l1, c2, a1, i1, c1),
            createKey(l1, c2, a1, i1b, c1),
            createKey(l1, c2, a1b, i1, c1),
            createKey(l1, c2, a1b, i1b, c1),
            createKey(l1b, c2, a1, i1, c1),
            createKey(l1b, c2, a1, i1b, c1),
            createKey(l1b, c2, a1b, i1, c1),
            createKey(l1b, c2, a1b, i1b, c1))
        .addEqualityGroup(
            createKey(l1, c2, a1, i2, c1),
            createKey(l1, c2, a1b, i2, c1),
            createKey(l1b, c2, a1, i2, c1),
            createKey(l1b, c2, a1b, i2, c1))
        .addEqualityGroup(
            createKey(l1, c2, a2, i1, c1),
            createKey(l1, c2, a2, i1b, c1),
            createKey(l1b, c2, a2, i1, c1),
            createKey(l1b, c2, a2, i1b, c1))
        .addEqualityGroup(
            createKey(l1, c2, a2, i2, c1),
            createKey(l1b, c2, a2, i2, c1))
        .addEqualityGroup(
            createKey(l1, c1, a1, i1, c2),
            createKey(l1, c1, a1, i1b, c2),
            createKey(l1, c1, a1b, i1, c2),
            createKey(l1, c1, a1b, i1b, c2),
            createKey(l1b, c1, a1, i1, c2),
            createKey(l1b, c1, a1, i1b, c2),
            createKey(l1b, c1, a1b, i1, c2),
            createKey(l1b, c1, a1b, i1b, c2))
        .addEqualityGroup(
            createKey(l1, c1, a1, i2, c2),
            createKey(l1, c1, a1b, i2, c2),
            createKey(l1b, c1, a1, i2, c2),
            createKey(l1b, c1, a1b, i2, c2))
        .addEqualityGroup(
            createKey(l1, c1, a2, i1, c2),
            createKey(l1, c1, a2, i1b, c2),
            createKey(l1b, c1, a2, i1, c2),
            createKey(l1b, c1, a2, i1b, c2))
        .addEqualityGroup(
            createKey(l1, c1, a2, i2, c2),
            createKey(l1b, c1, a2, i2, c2))
        .addEqualityGroup(
            createKey(l1, c2, a1, i1, c2),
            createKey(l1, c2, a1, i1b, c2),
            createKey(l1, c2, a1b, i1, c2),
            createKey(l1, c2, a1b, i1b, c2),
            createKey(l1b, c2, a1, i1, c2),
            createKey(l1b, c2, a1, i1b, c2),
            createKey(l1b, c2, a1b, i1, c2),
            createKey(l1b, c2, a1b, i1b, c2))
        .addEqualityGroup(
            createKey(l1, c2, a1, i2, c2),
            createKey(l1, c2, a1b, i2, c2),
            createKey(l1b, c2, a1, i2, c2),
            createKey(l1b, c2, a1b, i2, c2))
        .addEqualityGroup(
            createKey(l1, c2, a2, i1, c2),
            createKey(l1, c2, a2, i1b, c2),
            createKey(l1b, c2, a2, i1, c2),
            createKey(l1b, c2, a2, i1b, c2))
        .addEqualityGroup(
            createKey(l1, c2, a2, i2, c2),
            createKey(l1b, c2, a2, i2, c2))
        .addEqualityGroup(
            createKey(l2, c1, a1, i1, c1),
            createKey(l2, c1, a1, i1b, c1),
            createKey(l2, c1, a1b, i1, c1),
            createKey(l2, c1, a1b, i1b, c1))
        .addEqualityGroup(
            createKey(l2, c1, a1, i2, c1),
            createKey(l2, c1, a1b, i2, c1))
        .addEqualityGroup(
            createKey(l2, c1, a2, i1, c1),
            createKey(l2, c1, a2, i1b, c1))
        .addEqualityGroup(
            createKey(l2, c1, a2, i2, c1))
        .addEqualityGroup(
            createKey(l2, c2, a1, i1, c1),
            createKey(l2, c2, a1, i1b, c1),
            createKey(l2, c2, a1b, i1, c1),
            createKey(l2, c2, a1b, i1b, c1))
        .addEqualityGroup(
            createKey(l2, c2, a1, i2, c1),
            createKey(l2, c2, a1b, i2, c1))
        .addEqualityGroup(
            createKey(l2, c2, a2, i1, c1),
            createKey(l2, c2, a2, i1b, c1))
        .addEqualityGroup(
            createKey(l2, c2, a2, i2, c1))
        .addEqualityGroup(
            createKey(l2, c1, a1, i1, c2),
            createKey(l2, c1, a1, i1b, c2),
            createKey(l2, c1, a1b, i1, c2),
            createKey(l2, c1, a1b, i1b, c2))
        .addEqualityGroup(
            createKey(l2, c1, a1, i2, c2),
            createKey(l2, c1, a1b, i2, c2))
        .addEqualityGroup(
            createKey(l2, c1, a2, i1, c2),
            createKey(l2, c1, a2, i1b, c2))
        .addEqualityGroup(
            createKey(l2, c1, a2, i2, c2))
        .addEqualityGroup(
            createKey(l2, c2, a1, i1, c2),
            createKey(l2, c2, a1, i1b, c2),
            createKey(l2, c2, a1b, i1, c2),
            createKey(l2, c2, a1b, i1b, c2))
        .addEqualityGroup(
            createKey(l2, c2, a1, i2, c2),
            createKey(l2, c2, a1b, i2, c2))
        .addEqualityGroup(
            createKey(l2, c2, a2, i1, c2),
            createKey(l2, c2, a2, i1b, c2))
        .addEqualityGroup(
            createKey(l2, c2, a2, i2, c2))
        .addEqualityGroup(
            createDerivedKey(l1, c1, a1, i1, c1, a2, i2, c2),
            createDerivedKey(l1, c1, a1, i1b, c1, a2, i2, c2)
        )
        .addEqualityGroup(
            createDerivedKey(l1, c1, a2, i1, c1, a1, i2, c2),
            createDerivedKey(l1, c1, a2, i1b, c1, a1, i2, c2)
        )
        .testEquals();
  }

  private static SkyKey createKey(
      Label label, BuildConfiguration baseConfiguration, NativeAspectClass aspectClass,
      AspectParameters parameters, BuildConfiguration aspectConfiguration) {
    return AspectValueKey.createAspectKey(
        label,
        baseConfiguration,
        new AspectDescriptor(aspectClass, parameters),
        aspectConfiguration);
  }

  private static SkyKey createDerivedKey(
      Label label, BuildConfiguration baseConfiguration,
      NativeAspectClass aspectClass1, AspectParameters parameters1,
      BuildConfiguration aspectConfiguration1,
      NativeAspectClass aspectClass2, AspectParameters parameters2,
      BuildConfiguration aspectConfiguration2) {
    AspectKey baseKey =
        AspectValueKey.createAspectKey(
            label,
            baseConfiguration,
            new AspectDescriptor(aspectClass1, parameters1),
            aspectConfiguration1);
    return AspectValueKey.createAspectKey(
        label,
        baseConfiguration,
        ImmutableList.of(baseKey),
        new AspectDescriptor(aspectClass2, parameters2),
        aspectConfiguration2);
  }

}
