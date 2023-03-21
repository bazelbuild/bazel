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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyInterner;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyPool;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for the SkyKey class, checking hash code transience logic. */
@RunWith(JUnit4.class)
public final class SkyKeyTest {

  @Test
  @SuppressWarnings("ReturnValueIgnored") // Testing interactions with spy.
  public void testHashCodeTransience() throws Exception {
    // Given a freshly constructed HashCodeSpy object,
    HashCodeSpy hashCodeSpy = new HashCodeSpy();
    assertThat(hashCodeSpy.numberOfTimesHashCodeCalled).isEqualTo(0);

    // When a SkyKey is constructed with that HashCodeSpy as its argument,
    SkyKey originalKey = new HashCodeSpyKey(hashCodeSpy);

    // Then the HashCodeSpy reports that its hashcode method was called once.
    assertThat(hashCodeSpy.numberOfTimesHashCodeCalled).isEqualTo(1);

    // When the SkyKey's hashCode method is called,
    originalKey.hashCode();

    // Then the spy's hashCode method isn't called, because the SkyKey's hashCode was cached.
    assertThat(hashCodeSpy.numberOfTimesHashCodeCalled).isEqualTo(1);

    // When that SkyKey is serialized and then deserialized,
    SkyKey newKey =
        (SkyKey)
            TestUtils.fromBytes(
                new DeserializationContext(ImmutableClassToInstanceMap.of()),
                TestUtils.toBytes(
                    new SerializationContext(ImmutableClassToInstanceMap.of()), originalKey));

    // Then the new SkyKey recomputed its hashcode on deserialization.
    assertThat(newKey.hashCode()).isEqualTo(originalKey.hashCode());
    HashCodeSpy spyInNewKey = (HashCodeSpy) newKey.argument();
    assertThat(spyInNewKey.numberOfTimesHashCodeCalled).isEqualTo(1);

    // When the new SkyKey's hashCode method is called,
    newKey.hashCode();

    // Then the new SkyKey's spy's hashCode method is not called again.
    assertThat(spyInNewKey.numberOfTimesHashCodeCalled).isEqualTo(1);
  }

  static final class HashCodeSpy {
    private transient int numberOfTimesHashCodeCalled;

    @Override
    public int hashCode() {
      numberOfTimesHashCodeCalled++;
      return 42;
    }

    // Implemented so that numberOfTimesHashCodeCalled is not incremented when the debugger calls
    // toString() - the default Object#toString() calls hashCode().
    @Override
    public String toString() {
      return String.format("HashCodeSpy{count=%s}", numberOfTimesHashCodeCalled);
    }
  }

  @AutoCodec
  static final class HashCodeSpyKey extends AbstractSkyKey.WithCachedHashCode<HashCodeSpy> {

    HashCodeSpyKey(HashCodeSpy arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }
  }

  @Test
  public void skyKeyInterner_noGlobalPoolTestIntern() {
    SkyKey keyToIntern1 = SkyKeyForInternerTests.createInterned("HelloWorld");

    // Interning a duplicate instance will result the same instance to be returned.
    assertThat(SkyKeyForInternerTests.createInterned("HelloWorld")).isSameInstanceAs(keyToIntern1);
  }

  @Test
  public void skyKeyInterner_noGlobalPoolTestRemoval() {
    SkyKey keyToIntern1 = SkyKeyForInternerTests.createInterned("HelloWorld");

    assertThat(SkyKeyForInternerTests.createInterned("HelloWorld")).isSameInstanceAs(keyToIntern1);

    // Remove one instance from the interner and re-intern a duplicate one. The newly interned
    // instance is different from the previous one, which confirms that the previous interned
    // instance has already been successfully removed from the interner.
    keyToIntern1.getSkyKeyInterner().removeWeak(keyToIntern1);
    assertThat(SkyKeyForInternerTests.createInterned("HelloWorld"))
        .isNotSameInstanceAs(keyToIntern1);
  }

  @Test
  public void skyKeyInterner_withGlobalPool() {
    SkyKey keyToIntern1 = SkyKeyForInternerTests.createInterned("HelloWorld");
    SkyKey keyInPool = new SkyKeyForInternerTests("FooBar");

    SkyKeyPool globalPool =
        new SkyKeyPool() {
          @Override
          public SkyKey getOrWeakIntern(SkyKey key) {
            if (key.argument() == "FooBar") {
              return keyInPool;
            } else {
              return key.getSkyKeyInterner().weakIntern(key);
            }
          }

          @Override
          public void cleanupPool() {
            SkyKeyInterner.setGlobalPool(null);
          }
        };

    SkyKeyInterner.setGlobalPool(globalPool);

    assertThat(SkyKeyForInternerTests.createInterned("HelloWorld")).isSameInstanceAs(keyToIntern1);
    assertThat(SkyKeyForInternerTests.createInterned("FooBar")).isSameInstanceAs(keyInPool);

    globalPool.cleanupPool();
  }

  @After
  public void cleanUpGlobalPool() {
    SkyKeyInterner.setGlobalPool(null);
  }

  static final class SkyKeyForInternerTests extends AbstractSkyKey<String> {

    private static final SkyKeyInterner<SkyKeyForInternerTests> interner = SkyKey.newInterner();

    static SkyKeyForInternerTests createInterned(String arg) {
      return interner.intern(new SkyKeyForInternerTests(arg));
    }

    private SkyKeyForInternerTests(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }
  }
}
