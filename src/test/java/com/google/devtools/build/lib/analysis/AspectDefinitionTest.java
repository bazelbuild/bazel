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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Label;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for aspect definitions.
 */
@RunWith(JUnit4.class)
public class AspectDefinitionTest {
  /**
   * A dummy aspect factory. Is there to demonstrate how to define aspects and so that we can test
   * {@code attributeAspect}.
   */
  public static final class TestAspectFactory implements ConfiguredAspectFactory {
    private final AspectDefinition definition;

    /**
     * Normal aspects will have an argumentless constructor and their definition will be hard-wired
     * as a static member. This one is different so that we can create the definition in a test
     * method.
     */
    private TestAspectFactory(AspectDefinition definition) {
      this.definition = definition;
    }

    @Override
    public Aspect create(ConfiguredTarget base, RuleContext context, AspectParameters parameters) {
      throw new IllegalStateException();
    }

    @Override
    public AspectDefinition getDefinition() {
      return definition;
    }
  }

  @Test
  public void testSimpleAspect() throws Exception {
    new AspectDefinition.Builder("simple")
        .add(attr("$runtime", Type.LABEL).value(Label.parseAbsoluteUnchecked("//run:time")))
        .attributeAspect("deps", TestAspectFactory.class)
        .build();
  }

  @Test
  public void testAspectWithUserVisibleAttribute() throws Exception {
    try {
      new AspectDefinition.Builder("user_visible_attribute")
          .add(attr("invalid", Type.LABEL).value(Label.parseAbsoluteUnchecked("//run:time")))
          .attributeAspect("deps", TestAspectFactory.class)
          .build();
      fail(); // expected IllegalStateException
    } catch (IllegalStateException e) {
      // expected
    }
  }
}
