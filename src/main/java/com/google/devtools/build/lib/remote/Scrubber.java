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
package com.google.devtools.build.lib.remote;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.remote.RemoteScrubbing.Config;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.TextFormat;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/**
 * The {@link Scrubber} implements scrubbing of remote cache keys.
 *
 * <p>See the documentation for the {@code --experimental_remote_scrub_config} flag for more
 * information.
 */
public class Scrubber {

  /** An error that occurred while parsing the scrubbing configuration. */
  public static class ConfigParseException extends Exception {

    private ConfigParseException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  private final ImmutableList<SpawnScrubber> spawnScrubbers;

  @VisibleForTesting
  Scrubber(Config configProto) {
    ArrayList<SpawnScrubber> spawnScrubbers = new ArrayList<>();
    for (Config.Rule ruleProto : configProto.getRulesList()) {
      spawnScrubbers.add(new SpawnScrubber(ruleProto));
    }
    // Reverse the order so that later rules supersede earlier ones.
    Collections.reverse(spawnScrubbers);
    this.spawnScrubbers = ImmutableList.copyOf(spawnScrubbers);
  }

  /**
   * Constructs a {@link Scrubber} from the given configuration file, which must contain a {@link
   * Config} protocol buffer in text format.
   */
  public static Scrubber parse(String configPath) throws ConfigParseException {
    try (BufferedReader reader = Files.newBufferedReader(Paths.get(configPath))) {
      var builder = Config.newBuilder();
      TextFormat.getParser().merge(reader, builder);
      return new Scrubber(builder.build());
    } catch (IOException e) {
      throw new ConfigParseException(e.getMessage(), e);
    } catch (PatternSyntaxException e) {
      throw new ConfigParseException(
          String.format("in regex '%s': %s", e.getPattern(), e.getMessage()), e);
    }
  }

  /**
   * Returns a {@link SpawnScrubber} suitable for a {@link Spawn}, or {@code null} if the spawn does
   * not need to be scrubbed.
   */
  @Nullable
  public SpawnScrubber forSpawn(Spawn spawn) {
    for (SpawnScrubber spawnScrubber : spawnScrubbers) {
      if (spawnScrubber.matches(spawn)) {
        return spawnScrubber;
      }
    }
    return null;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof Scrubber scrubber)) {
      return false;
    }
    return spawnScrubbers.equals(scrubber.spawnScrubbers);
  }

  @Override
  public int hashCode() {
    return spawnScrubbers.hashCode();
  }

  /**
   * Encapsulates a set of transformations required to scrub the remote cache key for a set of
   * spawns.
   */
  public static class SpawnScrubber {

    private final Pattern mnemonicPattern;
    private final Pattern labelPattern;
    private final Pattern kindPattern;
    private final boolean matchTools;

    private final ImmutableList<Pattern> omittedInputPatterns;
    private final ImmutableMap<Pattern, String> argReplacements;
    private final String salt;

    private SpawnScrubber(Config.Rule ruleProto) {
      Config.Matcher matcherProto = ruleProto.getMatcher();
      this.mnemonicPattern = Pattern.compile(emptyToAll(matcherProto.getMnemonic()));
      this.labelPattern = Pattern.compile(emptyToAll(matcherProto.getLabel()));
      this.kindPattern = Pattern.compile(emptyToAll(matcherProto.getKind()));
      this.matchTools = matcherProto.getMatchTools();

      Config.Transform transformProto = ruleProto.getTransform();
      this.omittedInputPatterns =
          transformProto.getOmittedInputsList().stream()
              .map(Pattern::compile)
              .collect(toImmutableList());
      this.argReplacements =
          transformProto.getArgReplacementsList().stream()
              .collect(toImmutableMap(r -> Pattern.compile(r.getSource()), r -> r.getTarget()));
      this.salt = ruleProto.getTransform().getSalt();
    }

    private String emptyToAll(String s) {
      return s.isEmpty() ? ".*" : s;
    }

    /** Whether this scrubber applies to the given {@link Spawn}. */
    private boolean matches(Spawn spawn) {
      String mnemonic = spawn.getMnemonic();
      ActionOwner actionOwner = spawn.getResourceOwner().getOwner();
      String label = actionOwner.getLabel().getCanonicalForm();
      String kind = actionOwner.getTargetKind();
      boolean isForTool = actionOwner.isBuildConfigurationForTool();

      return (!isForTool || matchTools)
          && mnemonicPattern.matcher(mnemonic).matches()
          && labelPattern.matcher(label).matches()
          && kindPattern.matcher(kind).matches();
    }

    /** Whether an input with the given exec-relative path should be omitted from the cache key. */
    public boolean shouldOmitInput(PathFragment execPath) {
      for (Pattern pattern : omittedInputPatterns) {
        if (pattern.matcher(execPath.getPathString()).matches()) {
          return true;
        }
      }
      return false;
    }

    /** Transforms a command line argument. */
    public String transformArgument(String arg) {
      for (Map.Entry<Pattern, String> entry : argReplacements.entrySet()) {
        Pattern pattern = entry.getKey();
        String replacement = entry.getValue();
        // Don't use Pattern#replaceFirst because it allows references to capture groups.
        Matcher m = pattern.matcher(arg);
        if (m.find()) {
          arg = arg.substring(0, m.start()) + replacement + arg.substring(m.end());
        }
      }
      return arg;
    }

    /** Returns the scrubbing salt. */
    public String getSalt() {
      return salt;
    }

    @Override
    public final boolean equals(Object o) {
      if (!(o instanceof SpawnScrubber that)) return false;

      return matchTools == that.matchTools
          && Objects.equals(mnemonicPattern, that.mnemonicPattern)
          && Objects.equals(labelPattern, that.labelPattern)
          && Objects.equals(kindPattern, that.kindPattern)
          && Objects.equals(omittedInputPatterns, that.omittedInputPatterns)
          && Objects.equals(argReplacements, that.argReplacements)
          && Objects.equals(salt, that.salt);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          mnemonicPattern,
          labelPattern,
          kindPattern,
          matchTools,
          omittedInputPatterns,
          argReplacements,
          salt);
    }
  }
}
