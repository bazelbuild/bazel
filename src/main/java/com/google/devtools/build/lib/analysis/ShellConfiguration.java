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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/** A configuration fragment that tells where the shell is. */
public class ShellConfiguration extends BuildConfiguration.Fragment {

  /**
   * A codec for {@link ShellConfiguration}.
   *
   * <p>It does not handle the Bazel version, but that's fine, because we don't want to serialize
   * anything in Bazel.
   *
   * <p>We cannot use {@code AutoCodec} because the field {@link #actionEnvironment} is a lambda.
   * That does not necessarily need to be the case, but it's there in support for
   * {@link BuildConfiguration.Fragment#setupActionEnvironment()}, which is slated to be removed.
   */
  public static final class Codec implements ObjectCodec<ShellConfiguration> {
    @Override
    public Class<? extends ShellConfiguration> getEncodedClass() {
      return ShellConfiguration.class;
    }

    @Override
    public void serialize(SerializationContext context, ShellConfiguration obj,
        CodedOutputStream codedOut) throws SerializationException, IOException {
      context.serialize(obj.shellExecutable, codedOut);
    }

    @Override
    public ShellConfiguration deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      PathFragment shellExecutable = context.deserialize(codedIn);
      return new ShellConfiguration(shellExecutable, NO_ACTION_ENV.fromOptions(null));
    }
  }

  private static final ImmutableMap<OS, PathFragment> OS_SPECIFIC_SHELL =
      ImmutableMap.<OS, PathFragment>builder()
          .put(OS.WINDOWS, PathFragment.create("c:/tools/msys64/usr/bin/bash.exe"))
          .put(OS.FREEBSD, PathFragment.create("/usr/local/bin/bash"))
          .build();

  private final PathFragment shellExecutable;
  private final ShellActionEnvironment actionEnvironment;

  public ShellConfiguration(PathFragment shellExecutable,
      ShellActionEnvironment actionEnvironment) {
    this.shellExecutable = shellExecutable;
    this.actionEnvironment = actionEnvironment;
  }

  public PathFragment getShellExecutable() {
    return shellExecutable;
  }

  @Override
  public void setupActionEnvironment(Map<String, String> builder) {
    actionEnvironment.setupActionEnvironment(this, builder);
  }

  /** An option that tells Bazel where the shell is. */
  @AutoCodec(strategy = AutoCodec.Strategy.PUBLIC_FIELDS)
  public static class Options extends FragmentOptions {
    @Option(
        name = "shell_executable",
        converter = PathFragmentConverter.class,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        help =
            "Absolute path to the shell executable for Bazel to use. If this is unset, but the "
                + "BAZEL_SH environment variable is set on the first Bazel invocation (that starts "
                + "up a Bazel server), Bazel uses that. If neither is set, Bazel uses a hard-coded "
                + "default path depending on the operating system it runs on (Windows: "
                + "c:/tools/msys64/usr/bin/bash.exe, FreeBSD: /usr/local/bin/bash, all others: "
                + "/bin/bash). Note that using a shell that is not compatible with bash may lead "
                + "to build failures or runtime failures of the generated binaries."
    )
    public PathFragment shellExecutable;

    @Override
    public Options getHost() {
      Options host = (Options) getDefault();
      host.shellExecutable = shellExecutable;
      return host;
    }
  }

  /**
   * Encapsulates the contributions of {@link ShellConfiguration} to the action environment.
   *
   * <p>This is done this way because we need the shell environment to be different between Bazel
   * and Blaze, but configuration fragments are handed out based on their classes, thus,
   * doing this with inheritance would be difficult. The good old "has-a instead of this-a" pattern.
   */
  public interface ShellActionEnvironment {
    void setupActionEnvironment(ShellConfiguration configuration, Map<String, String> builder);
  }

  /**
   * A factory for shell action environments.
   *
   * <p>This is necessary because the Bazel shell action environment depends on whether we use
   * strict action environments or not, but we cannot simply hardcode the dependency on that bit
   * here because it doesn't exist in Blaze. Thus, during configuration creation time, we call this
   * factory which returns an object, which, when called, updates the actual action environment.
   */
  public interface ShellActionEnvironmentFactory {
    ShellActionEnvironment fromOptions(BuildOptions options);
  }

  /** A {@link ShellConfiguration} that contributes nothing to the action environment. */
  public static final ShellActionEnvironmentFactory NO_ACTION_ENV =
      (BuildOptions options) -> (ShellConfiguration config, Map<String, String> builder) -> {};

  /** the part of {@link ShellConfiguration} that determines where the shell is. */
  public interface ShellExecutableProvider {
    PathFragment getShellExecutable(BuildOptions options);
  }

  /** A shell executable whose path is hard-coded. */
  public static ShellExecutableProvider hardcodedShellExecutable(String shell) {
    return (BuildOptions options) -> PathFragment.create(shell);
  }

  /** The loader for {@link ShellConfiguration}. */
  public static class Loader implements ConfigurationFragmentFactory {
    private final ShellExecutableProvider shellExecutableProvider;
    private final ShellActionEnvironmentFactory actionEnvironmentFactory;
    private final ImmutableSet<Class<? extends FragmentOptions>> requiredOptions;

    public Loader(ShellExecutableProvider shellExecutableProvider,
        ShellActionEnvironmentFactory actionEnvironmentFactory,
        Class<? extends FragmentOptions>... requiredOptions) {
      this.shellExecutableProvider = shellExecutableProvider;
      this.actionEnvironmentFactory = actionEnvironmentFactory;
      this.requiredOptions = ImmutableSet.copyOf(requiredOptions);
    }

    @Nullable
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions) {
        return new ShellConfiguration(
            shellExecutableProvider.getShellExecutable(buildOptions),
            actionEnvironmentFactory.fromOptions(buildOptions));
    }

    public static PathFragment determineShellExecutable(
        OS os, Options options, PathFragment defaultShell) {
      if (options.shellExecutable != null) {
        return options.shellExecutable;
      }

      // Honor BAZEL_SH env variable for backwards compatibility.
      String path = System.getenv("BAZEL_SH");
      if (path != null) {
        return PathFragment.create(path);
      }
      // TODO(ulfjack): instead of using the OS Bazel runs on, we need to use the exec platform,
      // which may be different for remote execution. For now, this can be overridden with
      // --shell_executable, so at least there's a workaround.
      PathFragment result = OS_SPECIFIC_SHELL.get(os);
      return result != null ? result : defaultShell;
    }

    @Override
    public Class<? extends Fragment> creates() {
      return ShellConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return requiredOptions;
    }
  }
}
