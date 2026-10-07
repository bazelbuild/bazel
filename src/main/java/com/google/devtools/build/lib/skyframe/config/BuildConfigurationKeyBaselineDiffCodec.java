// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.OptionsDiff;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.PlatformConfigurationProvider;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Codec for {@link BuildConfigurationKey} that serializes an options diff relative to a baseline.
 *
 * <p>Because keys are serialized as diffs relative to the baseline, keys from builds with different
 * baselines will serialize to identical bytes with the same cache key even if the configuration is
 * different. This is not a problem for correctness in Skyframe because Skycache adds the baseline
 * configuration checksum as part of the FrontierNodeVersion. This isolates SkyValues from builds
 * with different configurations while allowing the diff-based serialization optimization to save
 * space.
 *
 * <p>The wire format is a single {@code byte} of bit flags encoding {@code isNoConfig}, {@code
 * isExec}, {@code trimTestOptions}, and {@code isTopLevelPlatform}, followed by:
 *
 * <ul>
 *   <li>if {@code isNoConfig}: nothing; the key represents the no-config configuration (see {@link
 *       CommonOptions#noConfigOptions}) and the other flags are unset
 *   <li>otherwise: the platform {@link Label} (only when {@code !isTopLevelPlatform}), and the
 *       {@link OptionsDiff}
 * </ul>
 */
public class BuildConfigurationKeyBaselineDiffCodec
    extends DeferredObjectCodec<BuildConfigurationKey> {

  private static final int IS_EXEC_MASK = 1;
  private static final int TRIM_TEST_OPTIONS_MASK = 2;
  private static final int IS_TOP_LEVEL_PLATFORM_MASK = 4;
  private static final int IS_NO_CONFIG_MASK = 8;

  @Override
  public boolean autoRegister() {
    return false;
  }

  @Override
  public Class<BuildConfigurationKey> getEncodedClass() {
    return BuildConfigurationKey.class;
  }

  @Override
  public void serialize(
      SerializationContext context, BuildConfigurationKey obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    BuildOptions options = obj.getOptions();
    if (options.hasNoConfig()) {
      // The no-config configuration has no PlatformOptions and thus no baseline to diff against.
      // It only varies in the analysis phase flags inherited from the configuration it is reached
      // from (see CommonOptions#noConfigOptions), which aren't written: the reader derives them
      // from its own top-level options, just like the CoreOptions codec does for
      // --check_visibility. This is sound because a writer and a reader that share a
      // FrontierNodeVersion agree on all of these flags except --check_visibility, which is
      // excluded from the top-level configuration checksum and which the reader applies to its own
      // no-config configuration anyway. Variants that a transition introduces below the top level
      // collapse onto the reader's top-level variant.
      codedOut.writeRawByte((byte) IS_NO_CONFIG_MASK);
      return;
    }

    Label platformLabel = options.get(PlatformOptions.class).computeTargetPlatform();
    CoreOptions coreOptions = options.get(CoreOptions.class);

    PlatformConfigurationProvider provider =
        context.getDependency(PlatformConfigurationProvider.class);
    boolean isExec = coreOptions.getIsExec();
    boolean trimTestOptions = !options.contains(TestOptions.class);
    BuildOptions baselineOptions =
        provider.getBaseOptionsForPlatform(platformLabel, isExec, trimTestOptions);

    // isExec and trimTestOptions have to travel on the wire: they select which baseline the reader
    // resolves, and the reader cannot recompute them before it has the options it is reconstructing
    // from that baseline.
    //
    // The platform label is elided when it is this build's top-level target platform, and the
    // reader substitutes its own top-level platform in its place. That substitution is the point of
    // the optimization: two builds whose top-level platforms differ produce byte-identical
    // encodings for the corresponding keys, so they share Skycache entries. It is sound because
    // Skycache mixes the top-level configuration checksum into FrontierNodeVersion, so entries from
    // builds with genuinely different baselines are still isolated from each other.
    //
    // Exec configurations always write their label explicitly: their platform is the exec platform,
    // which is unrelated to the top-level target platform and must survive the round trip verbatim.
    boolean isTopLevelPlatform =
        !isExec && platformLabel.equals(provider.getTopLevelPlatformLabel());
    int flags =
        (isExec ? IS_EXEC_MASK : 0)
            | (trimTestOptions ? TRIM_TEST_OPTIONS_MASK : 0)
            | (isTopLevelPlatform ? IS_TOP_LEVEL_PLATFORM_MASK : 0);
    codedOut.writeRawByte((byte) flags);
    if (!isTopLevelPlatform) {
      context.serialize(platformLabel, codedOut);
    }

    context.serialize(OptionsDiff.diff(baselineOptions, options), codedOut);
  }

  @Override
  public DeferredObjectCodec.DeferredValue<BuildConfigurationKey> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    byte flags = codedIn.readRawByte();
    if ((flags & IS_NO_CONFIG_MASK) != 0) {
      // The writer elided the inherited analysis phase flags; derive them from our top-level
      // options.
      var topLevelOptions = context.getDependency(BuildOptions.class);
      return () -> BuildConfigurationKey.create(CommonOptions.noConfigOptions(topLevelOptions));
    }

    PlatformConfigurationProvider provider =
        context.getDependency(PlatformConfigurationProvider.class);
    boolean isExec = (flags & IS_EXEC_MASK) != 0;
    boolean trimTestOptions = (flags & TRIM_TEST_OPTIONS_MASK) != 0;
    boolean isTopLevelPlatform = (flags & IS_TOP_LEVEL_PLATFORM_MASK) != 0;
    var builder = new PlatformDiffBuilder(provider, isExec, trimTestOptions);
    if (isTopLevelPlatform) {
      // The writer elided the label because it matched its own top-level platform; substitute ours.
      builder.platformLabel = provider.getTopLevelPlatformLabel();
    } else {
      context.deserialize(codedIn, builder, PlatformDiffBuilder::setPlatformLabel);
    }
    context.deserialize(codedIn, builder, PlatformDiffBuilder::setDiff);
    return builder;
  }

  /** Reconstructs a key by re-applying the serialized diff to the local baseline. */
  private static class PlatformDiffBuilder
      implements DeferredObjectCodec.DeferredValue<BuildConfigurationKey> {
    private final PlatformConfigurationProvider provider;
    private final boolean isExec;
    private final boolean trimTestOptions;

    private Label platformLabel;
    private OptionsDiff diff;

    private PlatformDiffBuilder(
        PlatformConfigurationProvider provider, boolean isExec, boolean trimTestOptions) {
      this.provider = provider;
      this.isExec = isExec;
      this.trimTestOptions = trimTestOptions;
    }

    private static void setPlatformLabel(PlatformDiffBuilder builder, Object value) {
      builder.platformLabel = (Label) value;
    }

    private static void setDiff(PlatformDiffBuilder builder, Object value) {
      builder.diff = (OptionsDiff) value;
    }

    @Override
    public BuildConfigurationKey call() throws SerializationException {
      BuildOptions baselineValue =
          provider.getBaseOptionsForPlatform(platformLabel, isExec, trimTestOptions);
      return BuildConfigurationKey.create(OptionsDiff.applyDiff(baselineValue, diff));
    }
  }
}
