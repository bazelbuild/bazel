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
package com.google.devtools.build.lib.syntax;

import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ASTNode}.
 */
@RunWith(JUnit4.class)
public class ASTNodeTest {

  private ASTNode node;

  @Before
  public final void createNode() throws Exception  {
    node = new ASTNode() {
      @Override
      public String toString() {
        return null;
      }
      @Override
      public void accept(SyntaxTreeVisitor visitor) {
      }
    };
  }

  @Test
  public void testHashCodeNotSupported() {
    try {
      node.hashCode();
      fail();
    } catch (UnsupportedOperationException e) {
      // yes!
    }
  }

  @Test
  public void testEqualsNotSupported() {
    try {
      node.equals(null);
      fail();
    } catch (UnsupportedOperationException e) {
      // yes!
    }
  }

}
