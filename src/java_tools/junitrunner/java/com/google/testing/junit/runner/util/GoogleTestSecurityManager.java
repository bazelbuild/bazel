// Copyright 2004 The Bazel Authors. All Rights Reserved.
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

import com.google.common.annotations.VisibleForTesting;

import java.security.Permission;

/**
 * A security manager that prevents things that are dangerous or
 * bad in a testing environment. Currently prevents System.exit() and
 * System.setSecurityManager().
 *
 * <p>For simplicity this is a Java 1.1 style security manager, ignoring
 * the whole Permissions framework. This should be fine unless you
 * are testing code that itself manipulates SecurityManagers.
 */
public final class GoogleTestSecurityManager extends SecurityManager {
  private volatile boolean enabled = true;

  /** Prevent System.exit() from ever being called. */
  @Override public void checkExit(int code) {
    if (enabled) {
      throw new SecurityException("Test code should never call System.exit()");
    }
  }

  //
  // The code below overrides the Java2 security mechanism to allow
  // (almost) all requests. This is OK vis-a-vis the Java default
  // (which is to run with no security policy at all).
  //
  // The default Java security policy is to pass through to the
  // Permissions check mechanism, which in turn by default denies
  // things. We override all of that (in essence, disabling Java2
  // Permissions) and just allow everything.
  //

  /**
   * Cache a copy of the permission that System.setSecurityManager() checks.
   */
  private final RuntimePermission securityManagerPermission =
      new RuntimePermission("setSecurityManager");

  /** Allow everything but creation of security managers. */
  @Override public void checkPermission(Permission p) {
    if (enabled && securityManagerPermission.equals(p)) {
      throw new SecurityException("GoogleTestSecurityManager is not designed to handle other " +
          "security managers.");
    }
  }

  /** Allow everything. */
  @Override public void checkPermission(Permission p, java.lang.Object o) {
    return;
  }

  public boolean isEnabled() { return enabled; }
  
  /**
   * If {@code GoogleTestSecurityManager} is the current security manager,
   * uninstall it.
   */
  public static void uninstallIfInstalled() {
    SecurityManager securityManager = System.getSecurityManager();
    if (securityManager instanceof GoogleTestSecurityManager) {
      GoogleTestSecurityManager testSecurityManager = (GoogleTestSecurityManager) securityManager;
      boolean wasEnabled = testSecurityManager.isEnabled();
      
      try {
        testSecurityManager.setEnabled(false);
        System.setSecurityManager(null);
      } finally {
        testSecurityManager.setEnabled(wasEnabled);
      }
    }
  }

  /**
   * The security manager can be disabled by the test runner (or any other
   * class in the same package) to allow it to exit with a meaningful result
   * code.
   */
  @VisibleForTesting
  synchronized void setEnabled(boolean enabled) { this.enabled = enabled; }
}
