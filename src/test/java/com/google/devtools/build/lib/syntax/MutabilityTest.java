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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;
import com.google.devtools.build.lib.vfs.PathFragment;
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

  private void assertCheckMutableFailsBecauseFrozen(Freezable value, Mutability mutability) {
    MutabilityException expected =
        assertThrows(MutabilityException.class, () -> Mutability.checkMutable(value, mutability));
    assertThat(expected).hasMessageThat().contains("trying to mutate a frozen object");
  }

  @Test
  public void freeze() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);

    Mutability.checkMutable(dummy, mutability);
    mutability.freeze();
    assertCheckMutableFailsBecauseFrozen(dummy, mutability);
  }

  @Test
  public void tryWithResources() throws Exception {
    Mutability escapedMutability;
    DummyFreezable dummy;
    try (Mutability mutability = Mutability.create("test")) {
      dummy = new DummyFreezable(mutability);
      Mutability.checkMutable(dummy, mutability);
      escapedMutability = mutability;
    }
    assertCheckMutableFailsBecauseFrozen(dummy, escapedMutability);
  }

  @Test
  public void queryLockedState_InitiallyUnlocked() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);

    assertThat(mutability.isLocked(dummy)).isFalse();
    Mutability.checkMutable(dummy, mutability);
  }

  @Test
  public void queryLockedState_OneLocation() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location locA = Location.fromPathFragment(PathFragment.create("/a"));

    mutability.lock(dummy, locA);
    assertThat(mutability.isLocked(dummy)).isTrue();
    assertThat(mutability.getLockLocations(dummy)).containsExactly(locA);
  }

  @Test
  public void queryLockedState_ManyLocations() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location locA = Location.fromPathFragment(PathFragment.create("/a"));
    Location locB = Location.fromPathFragment(PathFragment.create("/b"));
    Location locC = Location.fromPathFragment(PathFragment.create("/c"));
    Location locD = Location.fromPathFragment(PathFragment.create("/d"));

    mutability.lock(dummy, locA);
    mutability.lock(dummy, locB);
    mutability.lock(dummy, locC);
    mutability.lock(dummy, locD);
    assertThat(mutability.isLocked(dummy)).isTrue();
    assertThat(mutability.getLockLocations(dummy))
        .containsExactly(locA, locB, locC, locD).inOrder();
  }

  @Test
  public void queryLockedState_LockTwiceUnlockOnce() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location locA = Location.fromPathFragment(PathFragment.create("/a"));
    Location locB = Location.fromPathFragment(PathFragment.create("/b"));

    mutability.lock(dummy, locA);
    mutability.lock(dummy, locB);
    mutability.unlock(dummy, locA);
    assertThat(mutability.isLocked(dummy)).isTrue();
    assertThat(mutability.getLockLocations(dummy)).containsExactly(locB);
  }

  @Test
  public void queryLockedState_LockTwiceUnlockTwice() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location locA = Location.fromPathFragment(PathFragment.create("/a"));
    Location locB = Location.fromPathFragment(PathFragment.create("/b"));

    mutability.lock(dummy, locA);
    mutability.lock(dummy, locB);
    mutability.unlock(dummy, locA);
    mutability.unlock(dummy, locB);
    assertThat(mutability.isLocked(dummy)).isFalse();
    Mutability.checkMutable(dummy, mutability);
  }

  @Test
  public void cannotMutateLocked() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location locA = Location.fromPathFragment(PathFragment.create("/a"));
    Location locB = Location.fromPathFragment(PathFragment.create("/b"));

    mutability.lock(dummy, locA);
    mutability.lock(dummy, locB);
    MutabilityException expected =
        assertThrows(MutabilityException.class, () -> Mutability.checkMutable(dummy, mutability));
    assertThat(expected).hasMessageThat().contains(
        "trying to mutate a locked object (is it currently being iterated over by a for loop or "
            + "comprehension?)\nObject locked at the following location(s): /a:1, /b:1");
  }

  @Test
  public void unlockLocationMismatch() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location locA = Location.fromPathFragment(PathFragment.create("/a"));
    Location locB = Location.fromPathFragment(PathFragment.create("/b"));

    mutability.lock(dummy, locA);
    IllegalArgumentException expected =
        assertThrows(IllegalArgumentException.class, () -> mutability.unlock(dummy, locB));
    assertThat(expected).hasMessageThat().contains(
        "trying to unlock an object for a location at which it was not locked (/b:1)");
  }

  @Test
  public void lockAndThenFreeze() throws Exception {
    Mutability mutability = Mutability.create("test");
    DummyFreezable dummy = new DummyFreezable(mutability);
    Location loc = Location.fromPathFragment(PathFragment.create("/a"));

    mutability.lock(dummy, loc);
    mutability.freeze();
    assertThat(mutability.isLocked(dummy)).isFalse();
    // Should fail with frozen error, not locked error.
    assertCheckMutableFailsBecauseFrozen(dummy, mutability);
  }

  @Test
  public void wrongContext_CheckMutable() throws Exception {
    Mutability mutability1 = Mutability.create("test1");
    Mutability mutability2 = Mutability.create("test2");
    DummyFreezable dummy = new DummyFreezable(mutability1);

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class, () -> Mutability.checkMutable(dummy, mutability2));
    assertThat(expected).hasMessageThat().contains(
        "trying to mutate an object from a different context");
  }

  @Test
  public void wrongContext_Lock() throws Exception {
    Mutability mutability1 = Mutability.create("test1");
    Mutability mutability2 = Mutability.create("test2");
    DummyFreezable dummy = new DummyFreezable(mutability1);
    Location loc = Location.fromPathFragment(PathFragment.create("/a"));

    IllegalArgumentException expected =
        assertThrows(IllegalArgumentException.class, () -> mutability2.lock(dummy, loc));
    assertThat(expected).hasMessageThat().contains(
        "trying to lock an object from a different context");
  }

  @Test
  public void wrongContext_Unlock() throws Exception {
    Mutability mutability1 = Mutability.create("test1");
    Mutability mutability2 = Mutability.create("test2");
    DummyFreezable dummy = new DummyFreezable(mutability1);
    Location loc = Location.fromPathFragment(PathFragment.create("/a"));

    IllegalArgumentException expected =
        assertThrows(IllegalArgumentException.class, () -> mutability2.unlock(dummy, loc));
    assertThat(expected).hasMessageThat().contains(
        "trying to unlock an object from a different context");
  }

  @Test
  public void wrongContext_IsLocked() throws Exception {
    Mutability mutability1 = Mutability.create("test1");
    Mutability mutability2 = Mutability.create("test2");
    DummyFreezable dummy = new DummyFreezable(mutability1);
    Location loc = Location.fromPathFragment(PathFragment.create("/a"));

    mutability1.lock(dummy, loc);
    IllegalArgumentException expected =
        assertThrows(IllegalArgumentException.class, () -> mutability2.isLocked(dummy));
    assertThat(expected).hasMessageThat().contains(
        "trying to check the lock of an object from a different context");
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
