// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Mutability}. */
@RunWith(JUnit4.class)
public final class MutabilityTest {

  /** A trivial Freezable that can do nothing but freeze. */
  private static class DummyFreezable implements Mutability.Freezable {
    private final Mutability mutability;

    public DummyFreezable(Mutability mutability) {
      this.mutability = mutability;
    }

    @Override
    public Mutability mutability() {
      return mutability;
    }
  }

  private static void assertCheckMutableFailsBecauseFrozen(DummyFreezable x) {
    EvalException ex = assertThrows(EvalException.class, () -> Starlark.checkMutable(x));
    assertThat(ex).hasMessageThat().contains("trying to mutate a frozen DummyFreezable value");
  }

  @Test
  public void freeze() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);

    Starlark.checkMutable(dummy);
    mutability.freeze();
    assertCheckMutableFailsBecauseFrozen(dummy);
  }

  @Test
  public void tryWithResources() throws Exception {
    DummyFreezable dummy;
    try (Mutability mutability = Mutability.create("test")) {
      dummy = new DummyFreezable(mutability);
      Starlark.checkMutable(dummy);
    }
    assertCheckMutableFailsBecauseFrozen(dummy);
  }

  @Test
  public void initiallyMutable() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);

    Starlark.checkMutable(dummy);
  }

  @Test
  public void temporarilyImmutableDuringIteration() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable x = new DummyFreezable(mutability);
    x.updateIteratorCount(+1);
    EvalException ex = assertThrows(EvalException.class, () -> Starlark.checkMutable(x));
    assertThat(ex)
        .hasMessageThat()
        .contains("DummyFreezable value is temporarily immutable due to active for-loop iteration");

    x.updateIteratorCount(+1);
    x.updateIteratorCount(-1); // net +1 => still immutable
    ex = assertThrows(EvalException.class, () -> Starlark.checkMutable(x));
    assertThat(ex)
        .hasMessageThat()
        .contains("DummyFreezable value is temporarily immutable due to active for-loop iteration");

    x.updateIteratorCount(-1); // net 0 => mutable
    Starlark.checkMutable(x); // ok

    assertThrows(IllegalStateException.class, () -> x.updateIteratorCount(-1)); // underflow
  }

  @Test
  public void addIteratorAndThenFreeze() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    dummy.updateIteratorCount(+1);
    mutability.freeze();
    // Should fail with frozen error, not temporarily immutable error.
    assertCheckMutableFailsBecauseFrozen(dummy);
  }

  @Test
  public void checkUnsafeShallowFreezePrecondition_FailsWhenAlreadyFrozen() throws Exception {
    Mutability mutability = Mutability.create("test").freeze();
    assertThrows(
        IllegalArgumentException.class,
        () -> Freezable.checkUnsafeShallowFreezePrecondition(new DummyFreezable(mutability)));
  }

  @Test
  public void checkUnsafeShallowFreezePrecondition_FailsWhenDisallowed() throws Exception {
    Mutability mutability = Mutability.create("test");
    assertThrows(
        IllegalArgumentException.class,
        () -> Freezable.checkUnsafeShallowFreezePrecondition(new DummyFreezable(mutability)));
  }

  @Test
  public void checkUnsafeShallowFreezePrecondition_SucceedsWhenAllowed() throws Exception {
    Mutability mutability = Mutability.createAllowingShallowFreeze("test");
    Freezable.checkUnsafeShallowFreezePrecondition(new DummyFreezable(mutability));
  }
}
