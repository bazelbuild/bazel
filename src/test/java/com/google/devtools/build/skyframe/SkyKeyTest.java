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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.Serializable;

/**
 * Unit test for the SkyKey class, checking hash code transience logic.
 */
@RunWith(JUnit4.class)
public class SkyKeyTest {

  @Test
  public void testHashCodeTransience() throws Exception {
    // Given a freshly constructed HashCodeSpy object,
    HashCodeSpy hashCodeSpy = new HashCodeSpy();
    assertThat(hashCodeSpy.getNumberOfTimesHashCodeCalled()).isEqualTo(0);

    // When a SkyKey is constructed with that HashCodeSpy as its argument,
    SkyKey originalKey = SkyKey.create(SkyFunctionName.create("TEMP"), hashCodeSpy);

    // Then the HashCodeSpy reports that its hashcode method was called once.
    assertThat(hashCodeSpy.getNumberOfTimesHashCodeCalled()).isEqualTo(1);


    // When the SkyKey's hashCode method is called,
    originalKey.hashCode();

    // Then the spy's hashCode method isn't called, because the SkyKey's hashCode was cached.
    assertThat(hashCodeSpy.getNumberOfTimesHashCodeCalled()).isEqualTo(1);


    // When that SkyKey is serialized and then deserialized,
    SkyKey newKey = (SkyKey) TestUtils.deserializeObject(TestUtils.serializeObject(originalKey));

    // Then the new SkyKey's HashCodeSpy has not had its hashCode method called.
    HashCodeSpy spyInNewKey = (HashCodeSpy) newKey.argument();
    assertThat(spyInNewKey.getNumberOfTimesHashCodeCalled()).isEqualTo(0);


    // When the new SkyKey's hashCode method is called once,
    newKey.hashCode();

    // Then the new SkyKey's spy's hashCode method gets called.
    assertThat(spyInNewKey.getNumberOfTimesHashCodeCalled()).isEqualTo(1);


    // When the new SkyKey's hashCode method is called a second time,
    newKey.hashCode();

    // Then the new SkyKey's spy's hashCOde isn't called a second time, because the SkyKey's
    // hashCode was cached.
    assertThat(spyInNewKey.getNumberOfTimesHashCodeCalled()).isEqualTo(1);
  }

  private static class HashCodeSpy implements Serializable {
    private transient int numberOfTimesHashCodeCalled;

    public int getNumberOfTimesHashCodeCalled() {
      return numberOfTimesHashCodeCalled;
    }

    @Override
    public int hashCode() {
      numberOfTimesHashCodeCalled++;
      return 42;
    }
  }
}
