// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Function that reads the contents of a mapping file specified in {@code --platform_mappings} and
 * parses them for use in a {@link PlatformMappingValue}.
 *
 * <p>Note that this class only parses the mapping-file specific format, parsing (and validation) of
 * flags contained therein is left to the invocation of {@link
 * PlatformMappingValue#map(BuildConfigurationValue.Key, BuildOptions)}.
 */
public class PlatformMappingFunction implements SkyFunction {

  private final BlazeDirectories blazeDirectories;

  public PlatformMappingFunction(BlazeDirectories blazeDirectories) {
    this.blazeDirectories = blazeDirectories;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws PlatformMappingException, InterruptedException {
    PlatformMappingValue.Key platformMappingKey = (PlatformMappingValue.Key) skyKey.argument();
    PathFragment workspaceRelativeMappingPath =
        platformMappingKey.getWorkspaceRelativeMappingPath();

    Root workspaceRoot = Root.fromPath(blazeDirectories.getWorkspace());
    RootedPath rootedMappingPath =
        RootedPath.toRootedPath(workspaceRoot, workspaceRelativeMappingPath);
    FileValue fileValue = (FileValue) env.getValue(FileValue.key(rootedMappingPath));
    if (fileValue == null) {
      return null;
    }

    if (!fileValue.exists()) {
      if (!platformMappingKey.wasExplicitlySetByUser()) {
        // If no flag was passed and the default mapping file does not exist treat this as if the
        // mapping file was empty rather than an error.
        return PlatformMappingValue.EMPTY;
      }
      throw new PlatformMappingException(
          new MissingInputFileException(
              String.format(
                  "--platform_mappings was set to '%s' but no such file exists relative to the "
                      + "top-level workspace, '%s'",
                  workspaceRelativeMappingPath, workspaceRoot),
              Location.BUILTIN),
          SkyFunctionException.Transience.PERSISTENT);
    }
    if (fileValue.isDirectory()) {
      throw new PlatformMappingException(
          new MissingInputFileException(
              String.format(
                  "--platform_mappings was set to '%s' relative to the top-level workspace '%s' but"
                      + "that path refers to a directory, not a file",
                  workspaceRelativeMappingPath, workspaceRoot),
              Location.BUILTIN),
          SkyFunctionException.Transience.PERSISTENT);
    }

    Iterable<String> lines;
    try {
      lines =
          FileSystemUtils.readLines(fileValue.realRootedPath().asPath(), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new PlatformMappingException(e, SkyFunctionException.Transience.PERSISTENT);
    }

    return new Parser(lines.iterator()).parse().toPlatformMappingValue();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @VisibleForTesting
  static class PlatformMappingException extends SkyFunctionException {

    public PlatformMappingException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }

  @VisibleForTesting
  static class Parser {

    private final Iterator<String> lines;

    /**
     * Using an optional to represent the next line with contents, {@link Optional#empty()} if we
     * reached end of file.
     *
     * <p>Stores the current non-comment, non-empty, non-whitespace line. Don't access the field
     * directly, it can either be "used up" by calling {@link #consume()} or retrieved without
     * moving on by calling {@link #peek()}.
     */
    private Optional<String> line;

    Parser(Iterator<String> lines) {
      this.lines = lines;
    }

    Mappings parse() throws PlatformMappingException {
      goToNextContentLine();

      if (!line.isPresent()) {
        return new Mappings(ImmutableMap.of(), ImmutableMap.of());
      }

      Map<Label, Collection<String>> platformsToFlags = ImmutableMap.of();
      Map<Collection<String>, Label> flagsToPlatforms = ImmutableMap.of();

      if (!peek().equalsIgnoreCase("platforms:") && !peek().equalsIgnoreCase("flags:")) {
        throwParsingException("Expected 'platforms:' or 'flags:' but got " + peek());
      }

      if (peek().equalsIgnoreCase("platforms:")) {
        consume();
        platformsToFlags = platformsToFlags();
      }

      if (line.isPresent()) {
        if (!peek().equalsIgnoreCase("flags:")) {
          throwParsingException("Expected 'flags:' but got " + peek());
        }
        consume();
        flagsToPlatforms = flagsToPlatforms();
      }

      if (line.isPresent()) {
        throwParsingException("Expected end of file but got " + peek());
      }
      return new Mappings(platformsToFlags, flagsToPlatforms);
    }

    private Map<Label, Collection<String>> platformsToFlags() throws PlatformMappingException {
      ImmutableMap.Builder<Label, Collection<String>> platformsToFlags = ImmutableMap.builder();
      while (line.isPresent() && !peek().equalsIgnoreCase("flags:")) {
        Label platform = platform();
        Collection<String> flags = flags();
        platformsToFlags.put(platform, flags);
      }

      return platformsToFlags.build();
    }

    private Label platform() throws PlatformMappingException {
      if (!line.isPresent()) {
        throwParsingException("Expected platform label but got end of file");
      }
      String label = consume();

      Label platform;
      try {
        ImmutableMap<RepositoryName, RepositoryName> emptyRepositoryMapping = ImmutableMap.of();
        // It is ok for us to use an empty repository mapping in this instance because all platform
        // labels used in the mapping file should be relative to the root repository. Repository
        // mappings however only apply within a repository imported by the root repository.
        platform = Label.parseAbsolute(label, emptyRepositoryMapping);
      } catch (LabelSyntaxException e) {
        throw new PlatformMappingException(
            new PlatformMappingParsingException("Expected platform label but got " + label, e),
            SkyFunctionException.Transience.PERSISTENT);
      }
      return platform;
    }

    private Collection<String> flags() throws PlatformMappingException {
      ImmutableSet.Builder<String> flags = ImmutableSet.builder();
      // Note: Short form flags are not supported.
      while (lineIsFlag()) {
        flags.add(consume());
      }
      ImmutableSet<String> parsedFlags = flags.build();
      if (parsedFlags.isEmpty()) {
        if (!line.isPresent()) {
          throwParsingException("Expected a flag but got end of file");
        }
        throwParsingException(
            "Expected a standard format flag (starting with --) but got " + peek());
      }

      return parsedFlags;
    }

    private Map<Collection<String>, Label> flagsToPlatforms() throws PlatformMappingException {
      ImmutableMap.Builder<Collection<String>, Label> flagsToPlatforms = ImmutableMap.builder();
      while (lineIsFlag()) {
        Collection<String> flags = flags();
        Label platform = platform();
        flagsToPlatforms.put(flags, platform);
      }
      return flagsToPlatforms.build();
    }

    private String consume() {
      Preconditions.checkState(
          line.isPresent(), "Must make sure that a line exists before consuming.");
      String value = line.get();
      goToNextContentLine();
      return value;
    }

    private String peek() {
      Preconditions.checkState(
          line.isPresent(), "Must make sure that a line exists before peeking.");
      return line.get();
    }

    private void throwParsingException(String message) throws PlatformMappingException {
      throw new PlatformMappingException(
          new PlatformMappingParsingException(message), SkyFunctionException.Transience.PERSISTENT);
    }

    private boolean lineIsFlag() {
      return line.isPresent() && peek().startsWith("--");
    }

    private void goToNextContentLine() {
      while (lines.hasNext()) {
        String line = lines.next().trim();
        if (line.isEmpty() || line.startsWith("#")) {
          continue;
        }
        this.line = Optional.of(line);
        return;
      }
      line = Optional.empty();
    }
  }

  /**
   * Simple data holder to make testing easier. Only for use internal to this file/tests thereof.
   */
  @VisibleForTesting
  static class Mappings {
    final Map<Label, Collection<String>> platformsToFlags;
    final Map<Collection<String>, Label> flagsToPlatforms;

    Mappings(
        Map<Label, Collection<String>> platformsToFlags,
        Map<Collection<String>, Label> flagsToPlatforms) {
      this.platformsToFlags = platformsToFlags;
      this.flagsToPlatforms = flagsToPlatforms;
    }

    PlatformMappingValue toPlatformMappingValue() {
      return new PlatformMappingValue(platformsToFlags, flagsToPlatforms);
    }
  }

  /**
   * Inner wrapper exception to work around the fact that {@link SkyFunctionException} cannot carry
   * a message of its own.
   */
  private static class PlatformMappingParsingException extends Exception {
    public PlatformMappingParsingException(String message) {
      super(message);
    }

    public PlatformMappingParsingException(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
