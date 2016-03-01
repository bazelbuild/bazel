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

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects.AttributeAspect;
import com.google.devtools.build.lib.analysis.util.TestAspects.ExtraAttributeAspect;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.skyframe.AspectValue;

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
    Label l1 = Label.parseAbsolute("//a:l1");
    Label l1b = Label.parseAbsolute("//a:l1");
    Label l2 = Label.parseAbsolute("//a:l2");
    AspectParameters i1 = new AspectParameters.Builder()
        .addAttribute("foo", "bar")
        .build();
    AspectParameters i1b = new AspectParameters.Builder()
        .addAttribute("foo", "bar")
        .build();
    AspectParameters i2 = new AspectParameters.Builder()
        .addAttribute("foo", "baz")
        .build();
    NativeAspectClass<AttributeAspect> a1 =
        new NativeAspectClass<AttributeAspect>(AttributeAspect.class);
    NativeAspectClass<AttributeAspect> a1b =
        new NativeAspectClass<AttributeAspect>(AttributeAspect.class);
    NativeAspectClass<ExtraAttributeAspect> a2 =
        new NativeAspectClass<ExtraAttributeAspect>(ExtraAttributeAspect.class);

    // label: //a:l1 or //a:l2
    // aspectConfiguration: target or host
    // baseConfiguration: target or host
    // aspect: Attribute or ExtraAttribute
    // parameters: bar or baz

    new EqualsTester()
        .addEqualityGroup(
            AspectValue.key(l1, c1, c1, a1, i1),
            AspectValue.key(l1, c1, c1, a1, i1b),
            AspectValue.key(l1, c1, c1, a1b, i1),
            AspectValue.key(l1, c1, c1, a1b, i1b),
            AspectValue.key(l1b, c1, c1, a1, i1),
            AspectValue.key(l1b, c1, c1, a1, i1b),
            AspectValue.key(l1b, c1, c1, a1b, i1),
            AspectValue.key(l1b, c1, c1, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c1, a1, i2),
            AspectValue.key(l1, c1, c1, a1b, i2),
            AspectValue.key(l1b, c1, c1, a1, i2),
            AspectValue.key(l1b, c1, c1, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c1, a2, i1),
            AspectValue.key(l1, c1, c1, a2, i1b),
            AspectValue.key(l1b, c1, c1, a2, i1),
            AspectValue.key(l1b, c1, c1, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c1, a2, i2),
            AspectValue.key(l1b, c1, c1, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c2, a1, i1),
            AspectValue.key(l1, c1, c2, a1, i1b),
            AspectValue.key(l1, c1, c2, a1b, i1),
            AspectValue.key(l1, c1, c2, a1b, i1b),
            AspectValue.key(l1b, c1, c2, a1, i1),
            AspectValue.key(l1b, c1, c2, a1, i1b),
            AspectValue.key(l1b, c1, c2, a1b, i1),
            AspectValue.key(l1b, c1, c2, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c2, a1, i2),
            AspectValue.key(l1, c1, c2, a1b, i2),
            AspectValue.key(l1b, c1, c2, a1, i2),
            AspectValue.key(l1b, c1, c2, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c2, a2, i1),
            AspectValue.key(l1, c1, c2, a2, i1b),
            AspectValue.key(l1b, c1, c2, a2, i1),
            AspectValue.key(l1b, c1, c2, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c1, c2, a2, i2),
            AspectValue.key(l1b, c1, c2, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c1, a1, i1),
            AspectValue.key(l1, c2, c1, a1, i1b),
            AspectValue.key(l1, c2, c1, a1b, i1),
            AspectValue.key(l1, c2, c1, a1b, i1b),
            AspectValue.key(l1b, c2, c1, a1, i1),
            AspectValue.key(l1b, c2, c1, a1, i1b),
            AspectValue.key(l1b, c2, c1, a1b, i1),
            AspectValue.key(l1b, c2, c1, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c1, a1, i2),
            AspectValue.key(l1, c2, c1, a1b, i2),
            AspectValue.key(l1b, c2, c1, a1, i2),
            AspectValue.key(l1b, c2, c1, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c1, a2, i1),
            AspectValue.key(l1, c2, c1, a2, i1b),
            AspectValue.key(l1b, c2, c1, a2, i1),
            AspectValue.key(l1b, c2, c1, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c1, a2, i2),
            AspectValue.key(l1b, c2, c1, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c2, a1, i1),
            AspectValue.key(l1, c2, c2, a1, i1b),
            AspectValue.key(l1, c2, c2, a1b, i1),
            AspectValue.key(l1, c2, c2, a1b, i1b),
            AspectValue.key(l1b, c2, c2, a1, i1),
            AspectValue.key(l1b, c2, c2, a1, i1b),
            AspectValue.key(l1b, c2, c2, a1b, i1),
            AspectValue.key(l1b, c2, c2, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c2, a1, i2),
            AspectValue.key(l1, c2, c2, a1b, i2),
            AspectValue.key(l1b, c2, c2, a1, i2),
            AspectValue.key(l1b, c2, c2, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c2, a2, i1),
            AspectValue.key(l1, c2, c2, a2, i1b),
            AspectValue.key(l1b, c2, c2, a2, i1),
            AspectValue.key(l1b, c2, c2, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l1, c2, c2, a2, i2),
            AspectValue.key(l1b, c2, c2, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c1, a1, i1),
            AspectValue.key(l2, c1, c1, a1, i1b),
            AspectValue.key(l2, c1, c1, a1b, i1),
            AspectValue.key(l2, c1, c1, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c1, a1, i2),
            AspectValue.key(l2, c1, c1, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c1, a2, i1),
            AspectValue.key(l2, c1, c1, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c1, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c2, a1, i1),
            AspectValue.key(l2, c1, c2, a1, i1b),
            AspectValue.key(l2, c1, c2, a1b, i1),
            AspectValue.key(l2, c1, c2, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c2, a1, i2),
            AspectValue.key(l2, c1, c2, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c2, a2, i1),
            AspectValue.key(l2, c1, c2, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c1, c2, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c1, a1, i1),
            AspectValue.key(l2, c2, c1, a1, i1b),
            AspectValue.key(l2, c2, c1, a1b, i1),
            AspectValue.key(l2, c2, c1, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c1, a1, i2),
            AspectValue.key(l2, c2, c1, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c1, a2, i1),
            AspectValue.key(l2, c2, c1, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c1, a2, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c2, a1, i1),
            AspectValue.key(l2, c2, c2, a1, i1b),
            AspectValue.key(l2, c2, c2, a1b, i1),
            AspectValue.key(l2, c2, c2, a1b, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c2, a1, i2),
            AspectValue.key(l2, c2, c2, a1b, i2))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c2, a2, i1),
            AspectValue.key(l2, c2, c2, a2, i1b))
        .addEqualityGroup(
            AspectValue.key(l2, c2, c2, a2, i2))
        .testEquals();
  }
}
