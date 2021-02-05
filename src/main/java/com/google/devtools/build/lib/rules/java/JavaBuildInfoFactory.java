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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.java.WriteBuildInfoPropertiesAction.TimestampFormatter;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Java build info creation - generates properties file that contain the corresponding build-info
 * data.
 */
public abstract class JavaBuildInfoFactory implements BuildInfoFactory {
  public static final BuildInfoKey KEY = new BuildInfoKey("Java");

  static final PathFragment BUILD_INFO_NONVOLATILE_PROPERTIES_NAME =
      PathFragment.create("build-info-nonvolatile.properties");
  static final PathFragment BUILD_INFO_VOLATILE_PROPERTIES_NAME =
      PathFragment.create("build-info-volatile.properties");
  static final PathFragment BUILD_INFO_REDACTED_PROPERTIES_NAME =
      PathFragment.create("build-info-redacted.properties");

  private static final DateTimeFormatter DEFAULT_TIME_FORMAT =
      DateTimeFormatter.ofPattern("EEE MMM d HH:mm:ss yyyy");

  // A default formatter that returns a date in UTC format.
  @AutoCodec
  @VisibleForSerialization
  static class DefaultTimestampFormatter implements TimestampFormatter {
    @Override
    public String format(long timestamp) {
      return Instant.ofEpochMilli(timestamp).atZone(ZoneOffset.UTC).format(DEFAULT_TIME_FORMAT)
          + " ("
          + timestamp / 1000
          + ')';
    }
  }

  private static final TimestampFormatter DEFAULT_FORMATTER = new DefaultTimestampFormatter();

  @Override
  public final BuildInfoCollection create(
      BuildInfoContext context,
      BuildConfiguration config,
      Artifact stableStatus,
      Artifact volatileStatus) {
    WriteBuildInfoPropertiesAction redactedInfo =
        getHeader(
            context,
            config,
            BUILD_INFO_REDACTED_PROPERTIES_NAME,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            createRedactedTranslator(),
            true,
            true);
    WriteBuildInfoPropertiesAction nonvolatileInfo =
        getHeader(
            context,
            config,
            BUILD_INFO_NONVOLATILE_PROPERTIES_NAME,
            NestedSetBuilder.create(Order.STABLE_ORDER, stableStatus),
            createNonVolatileTranslator(),
            false,
            true);
    WriteBuildInfoPropertiesAction volatileInfo =
        getHeader(
            context,
            config,
            BUILD_INFO_VOLATILE_PROPERTIES_NAME,
            NestedSetBuilder.create(Order.STABLE_ORDER, volatileStatus),
            createVolatileTranslator(),
            true,
            false);
    List<Action> actions = new ArrayList<>(3);
    actions.add(redactedInfo);
    actions.add(nonvolatileInfo);
    actions.add(volatileInfo);
    return new BuildInfoCollection(
        actions,
        ImmutableList.of(nonvolatileInfo.getPrimaryOutput(), volatileInfo.getPrimaryOutput()),
        ImmutableList.of(redactedInfo.getPrimaryOutput()));
  }

  /** Creates a {@link BuildInfoPropertiesTranslator} to use for volatile keys. */
  protected abstract BuildInfoPropertiesTranslator createVolatileTranslator();

  /** Creates a {@link BuildInfoPropertiesTranslator} to use for non-volatile keys. */
  protected abstract BuildInfoPropertiesTranslator createNonVolatileTranslator();

  /**
   * Creates a {@link BuildInfoPropertiesTranslator} to use for redacted version of the build
   * informations.
   */
  protected abstract BuildInfoPropertiesTranslator createRedactedTranslator();

  /** Specifies the {@link TimestampFormatter} to use to output dates in the properties file. */
  protected TimestampFormatter getTimestampFormatter() {
    return DEFAULT_FORMATTER;
  }

  private WriteBuildInfoPropertiesAction getHeader(
      BuildInfoContext context,
      BuildConfiguration config,
      PathFragment propertyFileName,
      NestedSet<Artifact> inputs,
      BuildInfoPropertiesTranslator translator,
      boolean includeVolatile,
      boolean includeNonVolatile) {
    ArtifactRoot outputPath = config.getIncludeDirectory(RepositoryName.MAIN);
    final Artifact output =
        context.getBuildInfoArtifact(
            propertyFileName,
            outputPath,
            includeVolatile && !inputs.isEmpty()
                ? BuildInfoType.NO_REBUILD
                : BuildInfoType.FORCE_REBUILD_IF_CHANGED);
    return new WriteBuildInfoPropertiesAction(
        inputs, output, translator, includeVolatile, includeNonVolatile, getTimestampFormatter());
  }

  @Override
  public final BuildInfoKey getKey() {
    return KEY;
  }

  @Override
  public boolean isEnabled(BuildConfiguration config) {
    return config.hasFragment(JavaConfiguration.class);
  }
}
