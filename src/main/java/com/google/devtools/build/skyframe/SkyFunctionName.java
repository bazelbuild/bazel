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
package com.google.devtools.build.skyframe;

import com.google.common.base.Predicate;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import java.io.Serializable;
import java.util.Set;
import java.util.concurrent.ExecutionException;

/** An identifier for a {@code SkyFunction}. */
public final class SkyFunctionName implements Serializable {

  // In practice the number of unique SkyFunctionNames should be reasonably limited, use this cache
  // to avoid accidentally creating many of the same.
  private static final LoadingCache<String, SkyFunctionName> skyFunctionNameCache =
      CacheBuilder.newBuilder()
        .weakValues()
        .build(
            new CacheLoader<String, SkyFunctionName>() {
              @Override
              public SkyFunctionName load(String name) {
                return new SkyFunctionName(name);
              }
            });

  /**
   * A well-known key type intended for testing only. The associated SkyKey should have a String
   * argument.
   */
  // Needs to be after the cache is initialized.
  public static final SkyFunctionName FOR_TESTING = SkyFunctionName.create("FOR_TESTING");

  /** Create a SkyFunctionName identified by {@code name}. */
  public static SkyFunctionName create(String name) {
    try {
      return skyFunctionNameCache.get(name);
    } catch (ExecutionException e) {
      throw new IllegalStateException("Unexpected exception creating SkyFunctionName", e);
    }
  }

  private final String name;

  private SkyFunctionName(String name) {
    this.name = name;
  }

  public String getName() {
    return name;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SkyFunctionName)) {
      return false;
    }
    SkyFunctionName other = (SkyFunctionName) obj;
    return name.equals(other.name);
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIs(final SkyFunctionName functionName) {
    return new Predicate<SkyKey>() {
      @Override
      public boolean apply(SkyKey skyKey) {
        return functionName.equals(skyKey.functionName());
      }
    };
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIsIn(final Set<SkyFunctionName> functionNames) {
    return new Predicate<SkyKey>() {
      @Override
      public boolean apply(SkyKey skyKey) {
        return functionNames.contains(skyKey.functionName());
      }
    };
  }
}
