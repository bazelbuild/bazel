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
import static com.google.devtools.build.lib.server.FailureDetails.TargetPatterns.Code.DEPENDENCY_NOT_FOUND;
import static com.google.devtools.common.options.OptionsParser.STARLARK_SKIPPED_PREFIXES;
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
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.StarlarkOptionsParser;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.PlatformMappingValue.NativeAndStarlarkFlags;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
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

  private final ImmutableSet<Class<? extends FragmentOptions>> optionsClasses;

  PlatformMappingFunction(ImmutableSet<Class<? extends FragmentOptions>> optionsClasses) {
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

      Mappings parsed = parse(env, lines);
      if (parsed == null) {
        return null;
      }
      return parsed.toPlatformMappingValue(optionsClasses);
    }

    if (!platformMappingKey.wasExplicitlySetByUser()) {
      // If no flag was passed and the default mapping file does not exist treat this as if the
      // mapping file was empty rather than an error.
      return new PlatformMappingValue(ImmutableMap.of(), ImmutableMap.of(), ImmutableSet.of());
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

  @VisibleForTesting
  static final class PlatformMappingException extends SkyFunctionException {

    PlatformMappingException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }

  /** Parses the given lines, returns null if not all Skyframe deps are ready. */
  @VisibleForTesting
  @Nullable
  static Mappings parse(Environment env, List<String> lines) throws PlatformMappingException {
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

    ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags = ImmutableMap.of();
    // Flags -> platform mapping doesn't support Starlark flags. If the need arises we could upgrade
    // this to NativeAndStarlarkFlags like above.
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms = ImmutableMap.of();

    if (it.peek().equalsIgnoreCase("platforms:")) {
      it.next();
      platformsToFlags = readPlatformsToFlags(it, env);
      if (platformsToFlags == null) {
        return null;
      }
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

  /**
   * Lets {@link StarlarkOptionsParser} convert flag names to {@link Target}s through a Skyframe
   * {@link PackageFunction} lookup.
   */
  private static class SkyframeTargetLoader implements StarlarkOptionsParser.BuildSettingLoader {
    private final Environment env;

    public SkyframeTargetLoader(Environment env) {
      this.env = env;
    }

    @Nullable
    @Override
    public Target loadBuildSetting(String name)
        throws InterruptedException, TargetParsingException {
      Label asLabel = Label.parseCanonicalUnchecked(name);
      SkyKey pkgKey = asLabel.getPackageIdentifier();
      PackageValue pkg = (PackageValue) env.getValue(pkgKey);
      if (pkg == null) {
        return null;
      }
      try {
        return pkg.getPackage().getTarget(asLabel.getName());
      } catch (NoSuchTargetException e) {
        throw new TargetParsingException(
            String.format("Failed to load %s", name), e, DEPENDENCY_NOT_FOUND);
      }
    }
  }

  /**
   * Converts a set of native and Starlark flag settings to a {@link NativeAndStarlarkFlags}, or
   * returns null if not all Skyframe deps are ready.
   */
  @Nullable
  private static NativeAndStarlarkFlags parseStarlarkFlags(
      ImmutableSet<String> rawFlags, Environment env) throws PlatformMappingException {
    ImmutableSet.Builder<String> nativeFlags = ImmutableSet.builder();
    ImmutableList.Builder<String> starlarkFlags = ImmutableList.builder();
    for (String flagSetting : rawFlags) {
      if (STARLARK_SKIPPED_PREFIXES.stream().noneMatch(flagSetting::startsWith)) {
        nativeFlags.add(flagSetting);
      } else {
        starlarkFlags.add(flagSetting);
      }
    }
    // The StarlarkOptionsParser needs a native options parser just to inject its Starlark flag
    // values. It doesn't actually parse anything with the native parser.
    OptionsParser fakeNativeParser = OptionsParser.builder().build();
    StarlarkOptionsParser starlarkFlagParser =
        StarlarkOptionsParser.newStarlarkOptionsParser(
            new SkyframeTargetLoader(env), fakeNativeParser);
    try {
      if (!starlarkFlagParser.parseGivenArgs(starlarkFlags.build())) {
        return null;
      }
      return NativeAndStarlarkFlags.create(
          nativeFlags.build(), fakeNativeParser.getStarlarkOptions());
    } catch (OptionsParsingException e) {
      throw new PlatformMappingException(e, Transience.PERSISTENT);
    }
  }

  /**
   * Returns a parsed {@code platform -> flags setting}, or null if not all Skyframe deps are ready
   */
  @Nullable
  private static ImmutableMap<Label, NativeAndStarlarkFlags> readPlatformsToFlags(
      PeekingIterator<String> it, Environment env) throws PlatformMappingException {
    ImmutableMap.Builder<Label, NativeAndStarlarkFlags> platformsToFlags = ImmutableMap.builder();
    boolean needSkyframeDeps = false;
    while (it.hasNext() && !it.peek().equalsIgnoreCase("flags:")) {
      Label platform = readPlatform(it);
      ImmutableSet<String> flags = readFlags(it);
      NativeAndStarlarkFlags parsedFlags = parseStarlarkFlags(flags, env);
      if (parsedFlags == null) {
        needSkyframeDeps = true;
      } else {
        platformsToFlags.put(platform, parsedFlags);
      }
    }

    if (needSkyframeDeps) {
      return null;
    }

    try {
      return platformsToFlags.buildOrThrow();
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
      return flagsToPlatforms.buildOrThrow();
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
      // TODO(https://github.com/bazelbuild/bazel/issues/17127): This should be passed throw the
      //   main repo mapping!
      return Label.parseCanonical(line);
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
    final ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags;
    final ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms;

    Mappings(
        ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags,
        ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms) {
      this.platformsToFlags = platformsToFlags;
      this.flagsToPlatforms = flagsToPlatforms;
    }

    PlatformMappingValue toPlatformMappingValue(
        ImmutableSet<Class<? extends FragmentOptions>> optionsClasses) {
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
