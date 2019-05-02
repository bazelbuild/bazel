// Copyright 2002 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import java.security.Permission;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test the GoogleTestSecurityManager. Most of the tests only works if the
 * security manager is actually installed; otherwise they just
 * pass without testing anything.
 */
@RunWith(JUnit4.class)
public class GoogleTestSecurityManagerTest {
  private SecurityManager previousSecurityManager;
  private GoogleTestSecurityManager installedSecurityManager;

  @Before
  public void setUp() throws Exception {
    previousSecurityManager = System.getSecurityManager();

    // These tests assume that there isn't already a GoogleTestSecurityManager
    // running.
    GoogleTestSecurityManager.uninstallIfInstalled();
  }

  @After
  public void tearDown() {
    if (installedSecurityManager != null) {
      installedSecurityManager.setEnabled(false);
    }
    if (System.getSecurityManager() != previousSecurityManager) {
      System.setSecurityManager(previousSecurityManager);
    }
  }

  private void installTestSecurityManager() {
    SecurityManager previousSecurityManager = System.getSecurityManager();
    assertThat(previousSecurityManager).isNull();

    installedSecurityManager = new GoogleTestSecurityManager();
    System.setSecurityManager(installedSecurityManager);
  }

  @Test
  public void testExit() {
    installTestSecurityManager();

    assertThrows(SecurityException.class, () -> System.exit(1));
  }

  @Test
  public void testSetSecurityManager() {
    installTestSecurityManager();

    assertThrows(SecurityException.class, () -> System.setSecurityManager(new SecurityManager()));
  }

  /**
   * test enabling/disabling the security manager.  This test does not require
   * that a GoogleTestSecurityManager be installed.
   */
  @Test
  public void testSecurityManagerEnabled() {
    // create a security manager to use, but don't install it.
    GoogleTestSecurityManager sm = new GoogleTestSecurityManager();

    assertThat(sm.isEnabled()).isTrue();
    assertThrows(SecurityException.class, () -> sm.checkExit(0));

    sm.setEnabled(false);
    assertThat(!sm.isEnabled()).isTrue();

    sm.checkExit(0); // should allow

    sm.setEnabled(true);
    assertThat(sm.isEnabled()).isTrue();
    assertThrows(SecurityException.class, () -> sm.checkExit(0));
  }

  @Test
  public void testUninstallIfInstalled_whenInstalled() {
    installTestSecurityManager();
    GoogleTestSecurityManager.uninstallIfInstalled();

    assertWithMessage("Security manager should be enabled")
        .that(installedSecurityManager.isEnabled())
        .isTrue();
    assertWithMessage("Security manager should be uninstalled")
        .that(System.getSecurityManager())
        .isNull();
  }
  
  @Test
  public void testUninstallIfInstalled_whenNoSecurityManagerInstalled() {
    GoogleTestSecurityManager.uninstallIfInstalled();

    assertWithMessage("No security manager should be uninstalled")
        .that(System.getSecurityManager())
        .isNull();
  }
  
  @Test
  public void testUninstallIfInstalled_whenOtherSecurityManagerInstalled() {
    PermissiveSecurityManager otherSecurityManager = new PermissiveSecurityManager();
    System.setSecurityManager(otherSecurityManager);
    GoogleTestSecurityManager.uninstallIfInstalled();

    assertThat(System.getSecurityManager()).isSameInstanceAs(otherSecurityManager);
    System.setSecurityManager(null);
  }

  /**
   * Security manager that allows anything.
   */
  private static class PermissiveSecurityManager extends SecurityManager {
    @Override public void checkPermission(Permission p) {
      return;
    }

    @Override public void checkPermission(Permission p, java.lang.Object o) {
      return;
    }
  }
}
