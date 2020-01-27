// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.testbed;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;

/**
 * This is a testbed for testing XML output functionality.
 */
@RunWith(Enclosed.class)
public class XmlOutputExercises {

  /**
   * A sample test class testing .compareTo()
   */
  public static class ComparabilityTest {
    private ExampleObject exampleObject;

    @Before
    public void setUp() throws Exception {
      exampleObject = new ExampleObject("example");
    }

    @Test
    public void compareToEqualInstance() throws Exception {
      ExampleObject test = new ExampleObject("example");
      assertThat(test).isEquivalentAccordingToCompareTo(exampleObject);
    }

    @Test
    public void compareToGreaterInstance() throws Exception {
      ExampleObject test = new ExampleObject("gxample");
      assertThat(test).isGreaterThan(exampleObject);
    }

    @Test
    public void compareToLessInstance() throws Exception {
      ExampleObject test = new ExampleObject("axample");
      assertThat(test).isLessThan(exampleObject);
    }
  }

  /**
   * A sample test class testing .equals() and .hashCode()
   */
  public static class EqualsHashCodeTest {
    private ExampleObject exampleObject;

    @Before
    public void setUp() throws Exception {
      exampleObject = new ExampleObject("example");
    }

    @Test
    public void testEquals() throws Exception {
      assertThat(new ExampleObject("example")).isEqualTo(exampleObject);
      assertThat(new ExampleObject("wrong")).isNotEqualTo(exampleObject);
    }

    @Test
    public void testHashCode() throws Exception {
      assertThat(exampleObject.hashCode()).isEqualTo("example".hashCode());
    }
  }

  /**
   * A sample test class testing .toString()
   */
  public static class OtherTests {
    private ExampleObject exampleObject;

    @Before
    public void setUp() throws Exception {
      exampleObject = new ExampleObject("example");
    }

    @Test
    public void testToString() {
      assertThat(exampleObject.toString()).isEqualTo("example");
    }
  }


  /**
   * A sample test class testing failures
   */
  public static class FailureTest {
    @Test
    public void testFail() {
      fail("This is an expected error. The test is supposed to fail.");
    }
  }
}
