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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.github.benmanes.caffeine.cache.Ticker;

/**
 * SystemMillisTicker is a Ticker which uses the unix epoch as its fixed reference point.
 *
 * <p>It is preferable to com.github.benmanes.caffeine.cache.Ticker.SystemTicker because that class
 * doesn't increment its time-source while the system is asleep, which isn't appropriate when
 * expiring tokens which have wall-time-based expiry policies.
 */
public class SystemMillisTicker implements Ticker {
  public static final SystemMillisTicker INSTANCE = new SystemMillisTicker();

  private SystemMillisTicker() {}

  @Override
  public long read() {
    return System.currentTimeMillis() * 1_000_000;
  }
}
