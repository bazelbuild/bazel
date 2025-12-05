package build.stack.devtools.build.constellate.fakebuildapi.config;

import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigGlobalLibraryApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;

/**
 * Fake implementation of {@link ConfigGlobalLibraryApi}.
 */
public class FakeConfigGlobalLibrary implements ConfigGlobalLibraryApi {

  @Override
  public ConfigurationTransitionApi transition(
      StarlarkCallable implementation,
      Sequence<?> inputs,
      Sequence<?> outputs,
      StarlarkThread thread) {
    return new FakeConfigurationTransition();
  }

  @Override
  public ConfigurationTransitionApi analysisTestTransition(
      Dict<?, ?> changedSettings, StarlarkThread thread) {
    return new FakeConfigurationTransition();
  }
}
