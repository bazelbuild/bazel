// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.concurrent.PooledInterner.Pool;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PooledInterner} class, with and without global pool set. */
@RunWith(JUnit4.class)
public final class PooledInternerTest {
  @Nullable private Pool<ObjectForInternerTests> pool = null;

  private final PooledInterner<ObjectForInternerTests> interner =
      new PooledInterner<>() {
        @Nullable
        @Override
        protected Pool<ObjectForInternerTests> getPool() {
          return pool;
        }
      };

  private ObjectForInternerTests createInterned(String arg) {
    return interner.intern(new ObjectForInternerTests(arg));
  }

  @Test
  public void pooledInternerInterner_noGlobalPoolTestIntern() {
    ObjectForInternerTests keyToIntern1 = createInterned(/* arg= */ "HelloWorld");

    // Interning a duplicate instance will result the same instance to be returned.
    assertThat(createInterned(/* arg= */ "HelloWorld")).isSameInstanceAs(keyToIntern1);
  }

  @Test
  public void pooledInternerInterner_noGlobalPoolTestRemoval() {
    ObjectForInternerTests keyToIntern1 = createInterned(/* arg= */ "HelloWorld");

    assertThat(createInterned(/* arg= */ "HelloWorld")).isSameInstanceAs(keyToIntern1);

    // Remove one instance from the interner and re-intern a duplicate one. The newly interned
    // instance is different from the previous one, which confirms that the previous interned
    // instance has already been successfully removed from the interner.
    interner.removeWeak(keyToIntern1);
    assertThat(createInterned(/* arg= */ "HelloWorld")).isNotSameInstanceAs(keyToIntern1);
  }

  @Test
  public void pooledInternerInterner_withGlobalPool() {
    ObjectForInternerTests keyInPool = new ObjectForInternerTests(/* arg= */ "FooBar");
    pool =
        sample -> {
          if (sample.arg.equals("FooBar")) {
            return keyInPool;
          } else {
            return interner.weakIntern(sample);
          }
        };

    // If interned instance already exists in the pool, expect to get the pooled instance.
    assertThat(createInterned(/* arg= */ "FooBar")).isSameInstanceAs(keyInPool);

    // If interned instance does not exist in the pool, expect it to be weak interned. So interning
    // a duplicate instance will result the same instance to be returned.
    ObjectForInternerTests keyToIntern1 = createInterned(/* arg= */ "HelloWorld");
    assertThat(createInterned(/* arg= */ "HelloWorld")).isSameInstanceAs(keyToIntern1);
  }

  static final class ObjectForInternerTests {
    private final String arg;

    private ObjectForInternerTests(String arg) {
      this.arg = arg;
    }

    @Override
    public int hashCode() {
      return arg.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof ObjectForInternerTests
          && arg.equals(((ObjectForInternerTests) obj).arg);
    }
  }
}
