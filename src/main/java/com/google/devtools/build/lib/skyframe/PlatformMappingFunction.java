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

import static com.google.common.base.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * Function that reads the contents of a mapping file specified in {@code --platform_mappings} and
 * parses them for use in a {@link PlatformMappingValue}.
 *
 * <p>Note that this class only parses the mapping-file specific format, parsing (and validation) of
 * flags contained therein is left to the invocation of {@link PlatformMappingValue#map}.
 */
final class PlatformMappingFunction implements SkyFunction {

  private final ImmutableList<Class<? extends FragmentOptions>> optionsClasses;

  PlatformMappingFunction(ImmutableList<Class<? extends FragmentOptions>> optionsClasses) {
    this.optionsClasses = checkNotNull(optionsClasses);
  }

  @Nullable
  @Override
  public PlatformMappingValue compute(SkyKey skyKey, Environment env)
      throws PlatformMappingException, InterruptedException {
    PlatformMappingValue.Key platformMappingKey = (PlatformMappingValue.Key) skyKey.argument();
    PathFragment workspaceRelativeMappingPath =
        platformMappingKey.getWorkspaceRelativeMappingPath();

    PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    if (pkgLocator == null) {
      return null;
    }

    ImmutableList<Root> pathEntries = pkgLocator.getPathEntries();
    for (Root root : pathEntries) {
      RootedPath rootedMappingPath = RootedPath.toRootedPath(root, workspaceRelativeMappingPath);
      FileValue fileValue = (FileValue) env.getValue(FileValue.key(rootedMappingPath));
      if (fileValue == null) {
        return null;
      }

      if (!fileValue.exists()) {
        continue;
      }
      if (fileValue.isDirectory()) {
        throw new PlatformMappingException(
            new MissingInputFileException(
                createFailureDetail(
                    String.format(
                        "--platform_mappings was set to '%s' relative to the top-level workspace"
                            + " '%s' but that path refers to a directory, not a file",
                        workspaceRelativeMappingPath, root),
                    Code.PLATFORM_MAPPINGS_FILE_IS_DIRECTORY),
                Location.BUILTIN),
            SkyFunctionException.Transience.PERSISTENT);
      }

      List<String> lines;
      try {
        lines = FileSystemUtils.readLines(fileValue.realRootedPath().asPath(), UTF_8);
      } catch (IOException e) {
        throw new PlatformMappingException(e, SkyFunctionException.Transience.TRANSIENT);
      }

      return parse(lines).toPlatformMappingValue(optionsClasses);
    }

    if (!platformMappingKey.wasExplicitlySetByUser()) {
      // If no flag was passed and the default mapping file does not exist treat this as if the
      // mapping file was empty rather than an error.
      return new PlatformMappingValue(ImmutableMap.of(), ImmutableMap.of(), ImmutableList.of());
    }
    throw new PlatformMappingException(
        new MissingInputFileException(
            createFailureDetail(
                String.format(
                    "--platform_mappings was set to '%s' but no such file exists relative to the "
                        + "package path roots, '%s'",
                    workspaceRelativeMappingPath, pathEntries),
                Code.PLATFORM_MAPPINGS_FILE_NOT_FOUND),
            Location.BUILTIN),
        SkyFunctionException.Transience.PERSISTENT);
  }

  private static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setBuildConfiguration(BuildConfiguration.newBuilder().setCode(detailedCode))
        .build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @VisibleForTesting
  static final class PlatformMappingException extends SkyFunctionException {

    PlatformMappingException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }

  @VisibleForTesting
  static Mappings parse(List<String> lines) throws PlatformMappingException {
    PeekingIterator<String> it =
        Iterators.peekingIterator(
            lines.stream()
                .map(String::trim)
                .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                .iterator());

    if (!it.hasNext()) {
      return new Mappings(ImmutableMap.of(), ImmutableMap.of());
    }

    if (!it.peek().equalsIgnoreCase("platforms:") && !it.peek().equalsIgnoreCase("flags:")) {
      throw parsingException("Expected 'platforms:' or 'flags:' but got " + it.peek());
    }

    ImmutableMap<Label, ImmutableSet<String>> platformsToFlags = ImmutableMap.of();
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms = ImmutableMap.of();

    if (it.peek().equalsIgnoreCase("platforms:")) {
      it.next();
      platformsToFlags = readPlatformsToFlags(it);
    }

    if (it.hasNext()) {
      String line = it.next();
      if (!line.equalsIgnoreCase("flags:")) {
        throw parsingException("Expected 'flags:' but got " + line);
      }
      flagsToPlatforms = readFlagsToPlatforms(it);
    }

    if (it.hasNext()) {
      throw parsingException("Expected end of file but got " + it.next());
    }
    return new Mappings(platformsToFlags, flagsToPlatforms);
  }

  private static ImmutableMap<Label, ImmutableSet<String>> readPlatformsToFlags(
      PeekingIterator<String> it) throws PlatformMappingException {
    ImmutableMap.Builder<Label, ImmutableSet<String>> platformsToFlags = ImmutableMap.builder();
    while (it.hasNext() && !it.peek().equalsIgnoreCase("flags:")) {
      Label platform = readPlatform(it);
      ImmutableSet<String> flags = readFlags(it);
      platformsToFlags.put(platform, flags);
    }

    try {
      return platformsToFlags.build();
    } catch (IllegalArgumentException e) {
      throw parsingException(
          "Got duplicate platform entries but each platform key must be unique", e);
    }
  }

  private static ImmutableMap<ImmutableSet<String>, Label> readFlagsToPlatforms(
      PeekingIterator<String> it) throws PlatformMappingException {
    ImmutableMap.Builder<ImmutableSet<String>, Label> flagsToPlatforms = ImmutableMap.builder();
    while (it.hasNext() && it.peek().startsWith("--")) {
      ImmutableSet<String> flags = readFlags(it);
      Label platform = readPlatform(it);
      flagsToPlatforms.put(flags, platform);
    }

    try {
      return flagsToPlatforms.build();
    } catch (IllegalArgumentException e) {
      throw parsingException("Got duplicate flags entries but each flags key must be unique", e);
    }
  }

  private static Label readPlatform(PeekingIterator<String> it) throws PlatformMappingException {
    if (!it.hasNext()) {
      throw parsingException("Expected platform label but got end of file");
    }

    String line = it.next();
    try {
      // It is ok for us to use an empty repository mapping in this instance because all platform
      // labels used in the mapping file should be relative to the root repository. Repository
      // mappings however only apply within a repository imported by the root repository.
      return Label.parseAbsolute(line, /*repositoryMapping=*/ ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw parsingException("Expected platform label but got " + line, e);
    }
  }

  private static ImmutableSet<String> readFlags(PeekingIterator<String> it)
      throws PlatformMappingException {
    ImmutableSet.Builder<String> flags = ImmutableSet.builder();
    // Note: Short form flags are not supported.
    while (it.hasNext() && it.peek().startsWith("--")) {
      flags.add(it.next());
    }
    ImmutableSet<String> parsedFlags = flags.build();
    if (parsedFlags.isEmpty()) {
      throw parsingException(
          it.hasNext()
              ? "Expected a standard format flag (starting with --) but got " + it.peek()
              : "Expected a flag but got end of file");
    }
    return parsedFlags;
  }

  private static PlatformMappingException parsingException(String message) {
    return parsingException(message, /*cause=*/ null);
  }

  private static PlatformMappingException parsingException(String message, Exception cause) {
    return new PlatformMappingException(
        new PlatformMappingParsingException(message, cause),
        SkyFunctionException.Transience.PERSISTENT);
  }

  /**
   * Simple data holder to make testing easier. Only for use internal to this file/tests thereof.
   */
  @VisibleForTesting
  static final class Mappings {
    final ImmutableMap<Label, ImmutableSet<String>> platformsToFlags;
    final ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms;

    Mappings(
        ImmutableMap<Label, ImmutableSet<String>> platformsToFlags,
        ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms) {
      this.platformsToFlags = platformsToFlags;
      this.flagsToPlatforms = flagsToPlatforms;
    }

    PlatformMappingValue toPlatformMappingValue(
        ImmutableList<Class<? extends FragmentOptions>> optionsClasses) {
      return new PlatformMappingValue(platformsToFlags, flagsToPlatforms, optionsClasses);
    }
  }

  /**
   * Inner wrapper exception to work around the fact that {@link SkyFunctionException} cannot carry
   * a message of its own.
   */
  private static final class PlatformMappingParsingException extends Exception {

    PlatformMappingParsingException(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
