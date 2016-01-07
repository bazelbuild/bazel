// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.pkgcache.ParseFailureListener;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternValue.PrepareDepsOfPatternSkyKeyOrException;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternsValue.TargetPatternSequence;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * PrepareDepsOfPatternsFunction ensures the graph loads targets matching the pattern sequence and
 * their transitive dependencies.
 */
public class PrepareDepsOfPatternsFunction implements SkyFunction {

  public static ImmutableList<SkyKey> getSkyKeys(SkyKey skyKey, EventHandler eventHandler) {
    TargetPatternSequence targetPatternSequence = (TargetPatternSequence) skyKey.argument();
    Iterable<PrepareDepsOfPatternSkyKeyOrException> keysMaybe =
        PrepareDepsOfPatternValue.keys(targetPatternSequence.getPatterns(),
            targetPatternSequence.getOffset());

    ImmutableList.Builder<SkyKey> skyKeyBuilder = ImmutableList.builder();
    boolean handlerIsParseFailureListener = eventHandler instanceof ParseFailureListener;
    for (PrepareDepsOfPatternSkyKeyOrException skyKeyOrException : keysMaybe) {
      try {
        skyKeyBuilder.add(skyKeyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        handleTargetParsingException(eventHandler, handlerIsParseFailureListener,
            skyKeyOrException.getOriginalPattern(), e);
      }
    }

    return skyKeyBuilder.build();
  }

  private static final Function<SkyKey, TargetPatternKey> SKY_TO_TARGET_PATTERN =
      new Function<SkyKey, TargetPatternKey>() {
        @Nullable
        @Override
        public TargetPatternKey apply(SkyKey skyKey) {
          return (TargetPatternKey) skyKey.argument();
        }
      };

  public static ImmutableList<TargetPatternKey> getTargetPatternKeys(
      ImmutableList<SkyKey> skyKeys) {
    return ImmutableList.copyOf(Iterables.transform(skyKeys, SKY_TO_TARGET_PATTERN));
  }

  /**
   * Given a {@link SkyKey} that contains a sequence of target patterns, when this function returns
   * {@link PrepareDepsOfPatternsValue}, then all targets matching that sequence, and those targets'
   * transitive dependencies, have been loaded.
   */
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    EventHandler eventHandler = env.getListener();
    ImmutableList<SkyKey> skyKeys = getSkyKeys(skyKey, eventHandler);

    Map<SkyKey, ValueOrException<TargetParsingException>> tokensByKey =
        env.getValuesOrThrow(skyKeys, TargetParsingException.class);
    if (env.valuesMissing()) {
      return null;
    }

    boolean handlerIsParseFailureListener = eventHandler instanceof ParseFailureListener;
    for (SkyKey key : skyKeys) {
      try {
        // The only exception type throwable by PrepareDepsOfPatternFunction is
        // TargetParsingException. Therefore all ValueOrException values in the map will either
        // be non-null or throw TargetParsingException when get is called.
        Preconditions.checkNotNull(tokensByKey.get(key).get());
      } catch (TargetParsingException e) {
        // If a target pattern can't be evaluated, notify the user of the problem and keep going.
        handleTargetParsingException(eventHandler, handlerIsParseFailureListener, key, e);
      }
    }

    return new PrepareDepsOfPatternsValue(getTargetPatternKeys(skyKeys));
  }

  private static void handleTargetParsingException(EventHandler eventHandler,
      boolean handlerIsParseFailureListener, SkyKey key, TargetParsingException e) {
    TargetPatternKey patternKey = (TargetPatternKey) key.argument();
    String rawPattern = patternKey.getPattern();
    handleTargetParsingException(eventHandler, handlerIsParseFailureListener, rawPattern, e);
  }

  private static void handleTargetParsingException(EventHandler eventHandler,
      boolean handlerIsParseFailureListener, String rawPattern, TargetParsingException e) {
    String errorMessage = e.getMessage();
    eventHandler.handle(Event.error("Skipping '" + rawPattern + "': " + errorMessage));
    if (handlerIsParseFailureListener) {
      ParseFailureListener parseListener = (ParseFailureListener) eventHandler;
      parseListener.parsingError(rawPattern, errorMessage);
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

}
