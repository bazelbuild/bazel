// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.collect.nestedset.NestedSet}.
 */
@RunWith(JUnit4.class)
public class NestedSetImplTest {
  @SafeVarargs
  private static NestedSetBuilder<String> nestedSetBuilder(String... directMembers) {
    NestedSetBuilder<String> builder = NestedSetBuilder.stableOrder();
    builder.addAll(Lists.newArrayList(directMembers));
    return builder;
  }

  @Test
  public void simple() {
    NestedSet<String> set = nestedSetBuilder("a").build();

    assertEquals(ImmutableList.of("a"), set.toList());
    assertFalse(set.isEmpty());
  }

  @Test
  public void flatToString() {
    assertEquals("{}", nestedSetBuilder().build().toString());
    assertEquals("{a}", nestedSetBuilder("a").build().toString());
    assertEquals("{a, b}", nestedSetBuilder("a", "b").build().toString());
  }

  @Test
  public void nestedToString() {
    NestedSet<String> b = nestedSetBuilder("b1", "b2").build();
    NestedSet<String> c = nestedSetBuilder("c1", "c2").build();

    assertEquals("{{b1, b2}, a}",
      nestedSetBuilder("a").addTransitive(b).build().toString());
    assertEquals("{{b1, b2}, {c1, c2}, a}",
      nestedSetBuilder("a").addTransitive(b).addTransitive(c).build().toString());

    assertEquals("{b1, b2}", nestedSetBuilder().addTransitive(b).build().toString());
  }

  @Test
  public void isEmpty() {
    NestedSet<String> triviallyEmpty = nestedSetBuilder().build();
    assertTrue(triviallyEmpty.isEmpty());

    NestedSet<String> emptyLevel1 = nestedSetBuilder().addTransitive(triviallyEmpty).build();
    assertTrue(emptyLevel1.isEmpty());

    NestedSet<String> emptyLevel2 = nestedSetBuilder().addTransitive(emptyLevel1).build();
    assertTrue(emptyLevel2.isEmpty());

    NestedSet<String> triviallyNonEmpty = nestedSetBuilder("mango").build();
    assertFalse(triviallyNonEmpty.isEmpty());

    NestedSet<String> nonEmptyLevel1 = nestedSetBuilder().addTransitive(triviallyNonEmpty).build();
    assertFalse(nonEmptyLevel1.isEmpty());

    NestedSet<String> nonEmptyLevel2 = nestedSetBuilder().addTransitive(nonEmptyLevel1).build();
    assertFalse(nonEmptyLevel2.isEmpty());
  }

  @Test
  public void canIncludeAnyOrderInStableOrderAndViceVersa() {
    NestedSetBuilder.stableOrder()
        .addTransitive(NestedSetBuilder.compileOrder()
            .addTransitive(NestedSetBuilder.stableOrder().build()).build())
        .addTransitive(NestedSetBuilder.linkOrder()
            .addTransitive(NestedSetBuilder.stableOrder().build()).build())
        .addTransitive(NestedSetBuilder.naiveLinkOrder()
            .addTransitive(NestedSetBuilder.stableOrder().build()).build()).build();
    try {
      NestedSetBuilder.compileOrder().addTransitive(NestedSetBuilder.linkOrder().build()).build();
      fail("Shouldn't be able to include a non-stable order inside a different non-stable order!");
    } catch (IllegalArgumentException e) {
      // Expected.
    }
  }

  /**
   * A handy wrapper that allows us to use EqualsTester to test shallowEquals and shallowHashCode.
   */
  private static class SetWrapper<E> {
    NestedSet<E> set;

    SetWrapper(NestedSet<E> wrapped) {
      set = wrapped;
    }

    @Override
    public int hashCode() {
      return set.shallowHashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SetWrapper)) {
        return false;
      }
      try {
        @SuppressWarnings("unchecked")
        SetWrapper<E> other = (SetWrapper<E>) o;
        return set.shallowEquals(other.set);
      } catch (ClassCastException e) {
        return false;
      }
    }
  }

  @SafeVarargs
  private static <E> SetWrapper<E> flat(E... directMembers) {
    NestedSetBuilder<E> builder = NestedSetBuilder.stableOrder();
    builder.addAll(Lists.newArrayList(directMembers));
    return new SetWrapper<E>(builder.build());
  }

  @SafeVarargs
  private static <E> SetWrapper<E> nest(SetWrapper<E>... nested) {
    NestedSetBuilder<E> builder = NestedSetBuilder.stableOrder();
    for (SetWrapper<E> wrap : nested) {
      builder.addTransitive(wrap.set);
    }
    return new SetWrapper<E>(builder.build());
  }

  @SafeVarargs
  // Restricted to <Integer> to avoid ambiguity with the other nest() function.
  private static SetWrapper<Integer> nest(Integer elem, SetWrapper<Integer>... nested) {
    NestedSetBuilder<Integer> builder = NestedSetBuilder.stableOrder();
    builder.add(elem);
    for (SetWrapper<Integer> wrap : nested) {
      builder.addTransitive(wrap.set);
    }
    return new SetWrapper<>(builder.build());
  }

  @Test
  public void shallowEquality() {
    // Used below to check that inner nested sets can be compared by reference equality.
    SetWrapper<Integer> myRef = nest(nest(flat(7, 8)), flat(9));

    // Each "equality group" contains elements that are equal to one another
    // (according to equals() and hashCode()), yet distinct from all elements
    // of all other equality groups.
    new EqualsTester()
      .addEqualityGroup(flat(),
                        flat(),
                        nest(flat()))  // Empty set elision.
      .addEqualityGroup(NestedSetBuilder.<Integer>linkOrder().build())
      .addEqualityGroup(flat(3),
                        flat(3),
                        flat(3, 3))  // Element de-duplication.
      .addEqualityGroup(flat(4),
                        nest(flat(4))) // Automatic elision of one-element nested sets.
      .addEqualityGroup(NestedSetBuilder.<Integer>linkOrder().add(4).build())
      .addEqualityGroup(nestedSetBuilder("4").build())  // Like flat("4").
      .addEqualityGroup(flat(3, 4),
                        flat(3, 4))
      // Make a couple sets deep enough that shallowEquals() fails.
      // If this test case fails because you improve the representation, just delete it.
      .addEqualityGroup(nest(nest(flat(3, 4), flat(5)), nest(flat(6, 7), flat(8))))
      .addEqualityGroup(nest(nest(flat(3, 4), flat(5)), nest(flat(6, 7), flat(8))))
      .addEqualityGroup(nest(myRef),
                        nest(myRef),
                        nest(myRef, myRef))  // Set de-duplication.
      .addEqualityGroup(nest(3, myRef))
      .addEqualityGroup(nest(4, myRef))
      .testEquals();

    // Some things that are not tested by the above:
    //  - ordering among direct members
    //  - ordering among transitive sets
  }

  /** Checks that the builder always return a nested set with the correct order. */
  @Test
  public void correctOrder() {
    for (Order order : Order.values()) {
      for (int numDirects = 0; numDirects < 3; numDirects++) {
        for (int numTransitives = 0; numTransitives < 3; numTransitives++) {
          assertEquals(order, createNestedSet(order, numDirects, numTransitives, order).getOrder());
          // We allow mixing orders if one of them is stable. This tests that the top level order is
          // the correct one.
          assertEquals(order,
              createNestedSet(order, numDirects, numTransitives, Order.STABLE_ORDER).getOrder());
        }
      }
    }
  }

  private NestedSet<Integer> createNestedSet(Order order, int numDirects, int numTransitives,
      Order transitiveOrder) {
    NestedSetBuilder<Integer> builder = new NestedSetBuilder<>(order);

    for (int direct = 0; direct < numDirects; direct++) {
      builder.add(direct);
    }
    for (int transitive = 0; transitive < numTransitives; transitive++) {
      builder.addTransitive(new NestedSetBuilder<Integer>(transitiveOrder).add(transitive).build());
    }
    return builder.build();
  }
}
