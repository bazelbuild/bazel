package build.stack.devtools.build.constellate.fakebuildapi.config;

import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import net.starlark.java.eval.Printer;

/**
 * Fake implementation of {@link ConfigurationTransitionApi}.
 */
public class FakeConfigurationTransition implements ConfigurationTransitionApi {

  @Override
  public void repr(Printer printer) {}
}
