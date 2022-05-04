package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.RunEnvironmentInfoApi;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

@Immutable
public final class RunEnvironmentInfo extends NativeInfo implements RunEnvironmentInfoApi {

  /**
   * Singleton instance of the provider type for {@link DefaultInfo}.
   */
  public static final RunEnvironmentInfoProvider PROVIDER = new RunEnvironmentInfoProvider();

  private final ImmutableMap<String, String> environment;
  private final ImmutableList<String> inheritedEnvironment;
  private final boolean shouldErrorOnNonExecutableRule;

  /**
   * Constructs a new provider with the given fixed and inherited environment variables.
   */
  public RunEnvironmentInfo(Map<String, String> environment, List<String> inheritedEnvironment,
      boolean shouldErrorOnNonExecutableRule) {
    this.environment = ImmutableMap.copyOf(Preconditions.checkNotNull(environment));
    this.inheritedEnvironment =
        ImmutableList.copyOf(Preconditions.checkNotNull(inheritedEnvironment));
    this.shouldErrorOnNonExecutableRule = shouldErrorOnNonExecutableRule;
  }

  @Override
  public RunEnvironmentInfoProvider getProvider() {
    return PROVIDER;
  }

  /**
   * Returns environment variables which should be set when the target advertising this provider is
   * executed.
   */
  @Override
  public Map<String, String> getEnvironment() {
    return environment;
  }

  /**
   * Returns names of environment variables whose value should be inherited from the shell
   * environment when the target advertising this provider is executed.
   */
  @Override
  public ImmutableList<String> getInheritedEnvironment() {
    return inheritedEnvironment;
  }

  /**
   * Returns whether advertising this provider on a non-executable (and thus non-test) rule should
   * result in an error or a warning. The latter is required to not break testing.TestEnvironment,
   * which is now implemented via RunEnvironmentInfo.
   */
  public boolean shouldErrorOnNonExecutableRule() {
    return shouldErrorOnNonExecutableRule;
  }

  /**
   * Provider implementation for {@link RunEnvironmentInfoApi}.
   */
  public static class RunEnvironmentInfoProvider extends BuiltinProvider<RunEnvironmentInfo>
      implements RunEnvironmentInfoApi.RunEnvironmentInfoApiProvider {

    private RunEnvironmentInfoProvider() {
      super("RunEnvironmentInfo", RunEnvironmentInfo.class);
    }

    @Override
    public RunEnvironmentInfoApi constructor(Dict<?, ?> environment,
        Sequence<?> inheritedEnvironment) throws EvalException {
      return new RunEnvironmentInfo(
          Dict.cast(environment, String.class, String.class, "environment"),
          StarlarkList.immutableCopyOf(
              Sequence.cast(inheritedEnvironment, String.class, "inherited_environment")),
          /* shouldErrorOnNonExecutableRule */ true);
    }
  }
}
