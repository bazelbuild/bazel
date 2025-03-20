// Copyright 2021 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.dynamic;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ShrinkableSemaphore}. */
@RunWith(JUnit4.class)
public class ShrinkableSemaphoreTest {

  @Test
  public void testUpdateLoad_shrinksUnderLoad() {
    ShrinkableSemaphore sem = new ShrinkableSemaphore(12, 30, 0.75);
    assertThat(sem.availablePermits()).isEqualTo(12);
    sem.updateLoad(0);
    assertThat(sem.availablePermits()).isEqualTo(12);
    sem.updateLoad(10); // 1/3 load * 0.75 factor = 1/12 reduction
    assertThat(sem.availablePermits()).isEqualTo(11);
    sem.updateLoad(20);
    assertThat(sem.availablePermits()).isEqualTo(10);
    sem.updateLoad(30);
    assertThat(sem.availablePermits()).isEqualTo(9);
    sem.updateLoad(4);
    assertThat(sem.availablePermits()).isEqualTo(12);
  }

  @Test
  public void testUpdateLoad_shrinksProperlyWhenPermitsTaken() throws InterruptedException {
    ShrinkableSemaphore sem = new ShrinkableSemaphore(12, 30, 0.5);
    assertThat(sem.availablePermits()).isEqualTo(12);
    sem.acquire(5);
    sem.updateLoad(0);
    assertThat(sem.availablePermits()).isEqualTo(7);
    sem.updateLoad(10); // 1/3 load * 0.5 factor = 1/6 reduction, and 5 acquired
    assertThat(sem.availablePermits()).isEqualTo(5);
    sem.acquire(5);
    assertThat(sem.availablePermits()).isEqualTo(0);
    sem.updateLoad(20); // More permits temporarily taken than available
    assertThat(sem.availablePermits()).isEqualTo(-2);
    sem.release();
    assertThat(sem.availablePermits()).isEqualTo(-1);
    sem.updateLoad(30); // Only 6 permits allowed under load, 9 still acquired
    assertThat(sem.availablePermits()).isEqualTo(-3);
    sem.updateLoad(10); // Now 10 permits allowed under load, 9 still acquired
    assertThat(sem.availablePermits()).isEqualTo(1);
  }

  @Test
  public void testUpdateLoad_noShrinkWithZeroFactor() throws InterruptedException {
    ShrinkableSemaphore sem = new ShrinkableSemaphore(12, 30, 0);
    assertThat(sem.availablePermits()).isEqualTo(12);
    sem.acquire(5);
    sem.updateLoad(0);
    assertThat(sem.availablePermits()).isEqualTo(7);
    sem.updateLoad(30);
    assertThat(sem.availablePermits()).isEqualTo(7);
    sem.release(2);
    assertThat(sem.availablePermits()).isEqualTo(9);
  }

  @Test
  public void testUpdateLoad_noShrinkBelowZero() {
    ShrinkableSemaphore sem = new ShrinkableSemaphore(12, 30, 0.5);
    assertThat(sem.availablePermits()).isEqualTo(12);
    sem.updateLoad(60);
    assertThat(sem.availablePermits()).isEqualTo(1);
    sem.updateLoad(80);
    assertThat(sem.availablePermits()).isEqualTo(1);
    sem.updateLoad(40);
    assertThat(sem.availablePermits()).isEqualTo(4);
  }
}
