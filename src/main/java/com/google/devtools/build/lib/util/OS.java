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
package com.google.devtools.build.lib.util;

import java.util.EnumSet;

/**
 * Detects the running operating system and returns a describing enum value.
 */
public enum OS {
  DARWIN("osx", "Mac OS X"),
  FREEBSD("freebsd", "FreeBSD"),
  OPENBSD("openbsd", "OpenBSD"),
  LINUX("linux", "Linux"),
  WINDOWS("windows", "Windows"),
  UNKNOWN("unknown", "");

  private static final EnumSet<OS> POSIX_COMPATIBLE = EnumSet.of(DARWIN, FREEBSD, OPENBSD, LINUX);

  private final String canonicalName;
  private final String detectionName;

  OS(String canonicalName, String detectionName) {
    this.canonicalName = canonicalName;
    this.detectionName = detectionName;
  }

  public String getCanonicalName() {
    return canonicalName;
  }

  @Override
  public String toString() {
    return getCanonicalName();
  }

  private static final OS HOST_SYSTEM = determineCurrentOs();

  /**
   * The current operating system.
   */
  public static OS getCurrent() {
    return HOST_SYSTEM;
  }

  public static boolean isPosixCompatible() {
    return POSIX_COMPATIBLE.contains(getCurrent());
  }

  public static String getVersion() {
    return System.getProperty("os.version");
  }

  // We inject a the OS name through blaze.os, so we can have
  // some coverage for Windows specific code on Linux.
  private static OS determineCurrentOs() {
    String osName = System.getProperty("blaze.os");
    if (osName == null) {
      osName = System.getProperty("os.name");
    }

    if (osName == null) {
      return OS.UNKNOWN;
    }

    for (OS os : OS.values()) {
      // Windows have many names, all starting with "Windows".
      if (osName.startsWith(os.detectionName)) {
        return os;
      }
    }

    return OS.UNKNOWN;
  }
}
