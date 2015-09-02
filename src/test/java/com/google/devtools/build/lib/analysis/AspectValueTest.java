// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.syntax.Label;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.skyframe.AspectValue}.
 */
@RunWith(JUnit4.class)
public class AspectValueTest extends AnalysisTestCase {
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
  }

  @Override
  @After
  public void tearDown() throws Exception {
    super.tearDown();
  }

  @Test
  public void equality() throws Exception {
    update();
    BuildConfiguration c1 = getTargetConfiguration();
    BuildConfiguration c2 = getHostConfiguration();
    Label l1 = Label.parseAbsolute("//a:l1");
    Label l1b = Label.parseAbsolute("//a:l1");
    Label l2 = Label.parseAbsolute("//a:l2");
    AspectParameters i1 = new AspectParameters.Builder()
        .addAttribute("foo", "bar")
        .build();
    AspectParameters i2 = new AspectParameters.Builder()
        .addAttribute("foo", "baz")
        .build();
    Class<? extends ConfiguredAspectFactory> a1 = TestAspects.AttributeAspect.class;
    Class<? extends ConfiguredAspectFactory> a2 = TestAspects.ExtraAttributeAspect.class;

    new EqualsTester()
        .addEqualityGroup(AspectValue.key(l1, c1, a1, null), AspectValue.key(l1b, c1, a1, null))
        .addEqualityGroup(AspectValue.key(l1, c1, a1, i1))
        .addEqualityGroup(AspectValue.key(l1, c1, a1, i2))
        .addEqualityGroup(AspectValue.key(l2, c1, a1, null))
        .addEqualityGroup(AspectValue.key(l1, c2, a1, null))
        .addEqualityGroup(AspectValue.key(l2, c2, a1, null))
        .addEqualityGroup(AspectValue.key(l1, c1, a2, null))
        .addEqualityGroup(AspectValue.key(l2, c1, a2, null))
        .addEqualityGroup(AspectValue.key(l1, c2, a2, null))
        .addEqualityGroup(AspectValue.key(l2, c2, a2, null))
        .addEqualityGroup(l1)  // A random object
        .testEquals();
  }
}
