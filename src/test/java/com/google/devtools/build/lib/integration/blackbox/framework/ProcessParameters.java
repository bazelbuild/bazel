package com.google.devtools.build.lib.integration.blackbox.framework;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;

@AutoValue
public abstract class ProcessParameters {
  abstract String name();
  abstract ImmutableList<String> arguments();
  abstract File workingDirectory();
  abstract int expectedExitCode();
  abstract boolean expectedEmptyError();
  abstract Optional<ImmutableMap<String, String>> environment();
  abstract long timeoutMillis();
  abstract Optional<Path> redirectOutput();
  abstract Optional<Path> redirectError();

  public static Builder builder() {
    return new AutoValue_ProcessParameters.Builder()
        .setExpectedExitCode(0)
        .setExpectedEmptyError(true)
        .setTimeoutMillis(30 * 1000)
        .setArguments();
  }

  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setName(String value);
    public abstract Builder setArguments(String... args);
    public abstract Builder setArguments(ImmutableList<String> args);
    public abstract Builder setWorkingDirectory(File value);
    public abstract Builder setExpectedExitCode(int value);
    public abstract Builder setExpectedEmptyError(boolean value);
    public abstract Builder setEnvironment(ImmutableMap<String, String> map);
    public abstract Builder setTimeoutMillis(long millis);
    public abstract Builder setRedirectOutput(Path path);
    public abstract Builder setRedirectError(Path path);

    public abstract ProcessParameters build();
  }
}
