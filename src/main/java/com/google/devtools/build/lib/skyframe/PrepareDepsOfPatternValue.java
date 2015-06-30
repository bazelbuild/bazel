// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.List;

/**
 * The value returned by {@link PrepareDepsOfPatternFunction}. Because that function is
 * invoked only for its side effect (i.e. ensuring the graph contains targets matching the
 * pattern and its transitive dependencies), this value carries no information.
 *
 * <p>Because the returned value is always the same object, this value and the
 * {@link PrepareDepsOfPatternFunction} which computes it are incompatible with change pruning. It
 * should only be requested by consumers who do not require reevaluation when
 * {@link PrepareDepsOfPatternFunction} is reevaluated. Safe consumers include, e.g., top-level
 * consumers, and other functions which invoke {@link PrepareDepsOfPatternFunction} solely for its
 * side-effects.
 */
public class PrepareDepsOfPatternValue implements SkyValue {
  public static final PrepareDepsOfPatternValue INSTANCE = new PrepareDepsOfPatternValue();

  private PrepareDepsOfPatternValue() {
  }

  /**
   * Returns an iterable of {@link PrepareDepsOfPatternSkyKeyOrException}, with
   * {@link TargetPatternKey} arguments. If a provided pattern fails to parse, an element in the
   * returned iterable will throw when its
   * {@link PrepareDepsOfPatternSkyKeyOrException#getSkyKey} method is called and will return the
   * failing pattern when its {@link PrepareDepsOfPatternSkyKeyOrException#getOriginalPattern}
   * method is called.
   *
   * <p>There may be fewer returned elements than patterns provided as input. This function may
   * combine patterns to return an iterable of SkyKeys that is equivalent but more efficient to
   * evaluate, and will omit SkyKeys associated with negative patterns.
   *
   * @param patterns The list of patterns, e.g. "-foo/biz...". If a pattern's first character is
   *     "-", it is treated as a negative pattern.
   * @param policy The filtering policy, e.g. "only return test targets"
   * @param offset The offset to apply to relative target patterns.
   */
  @ThreadSafe
  public static Iterable<PrepareDepsOfPatternSkyKeyOrException> keys(List<String> patterns,
      FilteringPolicy policy, String offset) {
    Iterable<TargetPatternSkyKeyOrException> keysMaybe =
        TargetPatternValue.keys(patterns, policy, offset);
    ImmutableList.Builder<PrepareDepsOfPatternSkyKeyOrException> builder = ImmutableList.builder();
    for (TargetPatternSkyKeyOrException keyMaybe : keysMaybe) {
      try {
        SkyKey skyKey = keyMaybe.getSkyKey();
        if (!((TargetPatternKey) skyKey.argument()).isNegative()) {
          builder.add(new PrepareDepsOfPatternSkyKeyOrExceptionImpl(keyMaybe));
        }
      } catch (TargetParsingException e) {
        // keyMaybe.getSkyKey() may throw TargetParsingException if its corresponding pattern
        // failed to parse. If so, wrap the exception-holding TargetPatternSkyKeyOrException and
        // return it, so that our caller can deal with it.
        builder.add(new PrepareDepsOfPatternSkyKeyOrExceptionImpl(keyMaybe));
      }
    }
    return builder.build();
  }

  /**
   * Wrapper for a prepare deps of pattern {@link SkyKey} or the {@link TargetParsingException}
   * thrown when trying to create it.
   */
  public interface PrepareDepsOfPatternSkyKeyOrException {

    /**
     * Returns the stored {@link SkyKey} or throws {@link TargetParsingException} if one was thrown
     * when creating the key.
     */
    SkyKey getSkyKey() throws TargetParsingException;

    /**
     * Returns the pattern that resulted in the stored {@link SkyKey} or {@link
     * TargetParsingException}.
     */
    String getOriginalPattern();
  }

  /**
   * Converts from a {@link TargetPatternSkyKeyOrException} to a
   * {@link PrepareDepsOfPatternSkyKeyOrException}.
   */
  private static class PrepareDepsOfPatternSkyKeyOrExceptionImpl implements
      PrepareDepsOfPatternSkyKeyOrException {

    private final TargetPatternSkyKeyOrException wrapped;

    private PrepareDepsOfPatternSkyKeyOrExceptionImpl(TargetPatternSkyKeyOrException wrapped) {
      this.wrapped = wrapped;
    }

    @Override
    public SkyKey getSkyKey() throws TargetParsingException {
      return new SkyKey(SkyFunctions.PREPARE_DEPS_OF_PATTERN, wrapped.getSkyKey().argument());
    }

    @Override
    public String getOriginalPattern() {
      return wrapped.getOriginalPattern();
    }
  }
}
