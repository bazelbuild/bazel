// Copyright 2025 The Bazel Authors. All rights reserved.
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
package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.EqualsTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SymbolGenerator}. */
@RunWith(JUnit4.class)
public final class SymbolGeneratorTest {

  @Test
  public void localSymbol_equalityAndHashCode() {
    Object owner1 = new Object();
    Object owner2 = new Object();

    SymbolGenerator<Object> generator1 = SymbolGenerator.create(owner1);
    SymbolGenerator<Object> generator2 = SymbolGenerator.create(owner2);

    SymbolGenerator.Symbol<Object> s1a = generator1.generate(); // owner1, index 0
    SymbolGenerator.Symbol<Object> s1b = generator1.generate(); // owner1, index 1
    SymbolGenerator.Symbol<Object> s2a = generator2.generate(); // owner2, index 0

    // Create another generator with the same owner object
    SymbolGenerator<Object> generator1Prime = SymbolGenerator.create(owner1);
    SymbolGenerator.Symbol<Object> s1aPrime = generator1Prime.generate(); // owner1, index 0

    new EqualsTester()
        .addEqualityGroup(s1a, s1a) // Reflexive
        .addEqualityGroup(s1b)
        .addEqualityGroup(s2a)
        .addEqualityGroup(s1aPrime) // Different generator instance, same owner and index
        .testEquals();

    assertThat(s1a).isNotEqualTo(s1b);
    assertThat(s1a).isNotEqualTo(s2a);
    assertThat(s1b).isNotEqualTo(s2a);

    // These should not be equal because the index will be different
    assertThat(s1a).isNotEqualTo(s1aPrime);
  }

  @Test
  public void localSymbol_hashCode_isMemoized() {
    class HashCounter {
      private int hashCodeCalls = 0;
      private final int hash;

      HashCounter(int hash) {
        this.hash = hash;
      }

      @Override
      public int hashCode() {
        hashCodeCalls++;
        return hash;
      }
    }

    HashCounter owner = new HashCounter(123);
    SymbolGenerator<HashCounter> generator = SymbolGenerator.create(owner);
    SymbolGenerator.Symbol<HashCounter> symbol = generator.generate();

    int hash1 = symbol.hashCode();
    assertThat(owner.hashCodeCalls).isEqualTo(1);

    int hash2 = symbol.hashCode();
    assertThat(hash1).isEqualTo(hash2);
    assertThat(owner.hashCodeCalls).isEqualTo(1); // Should not have incremented

    int hash3 = symbol.hashCode();
    assertThat(hash1).isEqualTo(hash3);
    assertThat(owner.hashCodeCalls).isEqualTo(1); // Should still be 1
  }

  @Test
  public void globalSymbol_equalityAndHashCode() {
    Object owner1 = new Object();
    Object owner2 = new Object();

    SymbolGenerator<Object> generator1 = SymbolGenerator.create(owner1);
    SymbolGenerator<Object> generator2 = SymbolGenerator.create(owner2);

    SymbolGenerator.Symbol<Object> local1 = generator1.generate();
    SymbolGenerator.Symbol<Object> local2 = generator2.generate();

    SymbolGenerator.Symbol<Object> g1a = local1.exportAs("name1");
    SymbolGenerator.Symbol<Object> g1aDup = local1.exportAs("name1");
    SymbolGenerator.Symbol<Object> g1b = local1.exportAs("name2");
    SymbolGenerator.Symbol<Object> g2a = local2.exportAs("name1");

    new EqualsTester()
        .addEqualityGroup(g1a, g1aDup)
        .addEqualityGroup(g1b)
        .addEqualityGroup(g2a)
        .testEquals();
  }
}
