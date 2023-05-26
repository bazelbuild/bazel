// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.cquery;


import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Parents class for cquery output callbacks. Handles names and outputting contents of result list
 * that is populated by child classes.
 *
 * <p>Human-readable cquery outputters should output short configuration IDs via {@link
 * #shortId(BuildConfigurationValue)} for easier reading. Machine-readable output, which are more
 * focused on completeness, should output full configuration checksums.
 */
public abstract class CqueryThreadsafeCallback
    extends NamedThreadSafeOutputFormatterCallback<KeyedConfiguredTarget> {

  protected final ExtendedEventHandler eventHandler;
  protected final CqueryOptions options;
  protected OutputStream outputStream;
  protected Writer printStream;
  // Skyframe calls incur a performance cost, even on cache hits. Consider this before exposing
  // direct executor access to child classes.
  private final SkyframeExecutor skyframeExecutor;
  private final Map<BuildConfigurationKey, BuildConfigurationValue> configCache =
      new ConcurrentHashMap<>();
  protected final ConfiguredTargetAccessor accessor;

  private final List<String> result = new ArrayList<>();
  private final boolean uniquifyResults;

  @SuppressWarnings("DefaultCharset")
  CqueryThreadsafeCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<KeyedConfiguredTarget> accessor,
      boolean uniquifyResults) {
    this.eventHandler = eventHandler;
    this.options = options;
    if (out != null) {
      this.outputStream = out;
      // This code intentionally uses the platform default encoding.
      this.printStream = new BufferedWriter(new OutputStreamWriter(out));
    }
    this.skyframeExecutor = skyframeExecutor;
    this.accessor = (ConfiguredTargetAccessor) accessor;
    this.uniquifyResults = uniquifyResults;
  }

  public void addResult(String string) {
    result.add(string);
  }

  @VisibleForTesting
  public List<String> getResult() {
    return result;
  }

  @Override
  public void close(boolean failFast) throws InterruptedException, IOException {
    if (!failFast && printStream != null) {
      List<String> resultsToPrint = uniquifyResults ? ImmutableSet.copyOf(result).asList() : result;
      for (String s : resultsToPrint) {
        // TODO(ulfjack): We should use queryOptions.getLineTerminator() instead.
        printStream.append(s).append("\n");
      }
      printStream.flush();
    }
  }

  @Nullable
  protected BuildConfigurationValue getConfiguration(BuildConfigurationKey configKey) {
    // Experiments querying:
    //     cquery --output=graph "deps(//src:main/java/com/google/devtools/build/lib:runtime)"
    // 10 times on a warm Blaze instance show 7% less total query time when using this cache vs.
    // calling Skyframe directly (and relying on Skyframe's cache).
    if (configKey == null) {
      return null;
    }
    return configCache.computeIfAbsent(
        configKey, key -> skyframeExecutor.getConfiguration(eventHandler, key));
  }

  /**
   * Returns a user-friendly configuration identifier, using special IDs for null configurations.
   */
  protected static String shortId(@Nullable BuildConfigurationValue config) {
    return config == null ? "null" : config.shortId();
  }
}

