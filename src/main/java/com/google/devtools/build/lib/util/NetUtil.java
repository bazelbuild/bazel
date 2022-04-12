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

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.function.Supplier;

/**
 * Various utility methods for network related stuff.
 */
public final class NetUtil {

  private static String hostname = null;
  private static Supplier<String> hostnameSupplier = NetUtil::computeShortHostName;

  private NetUtil() {}

  /**
   * Returns the *cached* short hostname (computed at most once per the lifetime of a server). Can
   * take seconds to complete when the cache is cold.
   */
  public static String getCachedShortHostName() {
    if (hostname == null) {
      synchronized (NetUtil.class) {
        if (hostname == null) {
          hostname = firstNonNull(hostnameSupplier.get(), "unknown");
          hostnameSupplier = null;
        }
      }
    }
    return hostname;
  }

  /**
   * Sets a {@link Supplier} for the hostname to return from {@link #getCachedShortHostName}.
   *
   * <p>If not called, the hostname comes from {@link #computeShortHostName}. To prevent multiple
   * different hostnames from being used, it is illegal to call this after {@link
   * #getCachedShortHostName} has been called.
   */
  public static synchronized void overrideHostnameSupplier(Supplier<String> override) {
    checkState(hostname == null, "Hostname already set to %s", hostname);
    hostnameSupplier = checkNotNull(override);
  }

  /**
   * Returns the short hostname or <code>unknown</code> if the host name could not be determined.
   * Performs reverse DNS lookup and can take seconds to complete.
   */
  private static String computeShortHostName() {
    try {
      return InetAddress.getLocalHost().getHostName();
    } catch (UnknownHostException e) {
      return "unknown";
    }
  }
}
