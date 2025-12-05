package build.stack.devtools.build.constellate.fakebuildapi.config;

import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigStarlarkCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import build.stack.devtools.build.constellate.fakebuildapi.FakeProviderApi;

/**
 * Fake implementation of {@link ConfigStarlarkCommonApi}.
 */
public class FakeConfigStarlarkCommon implements ConfigStarlarkCommonApi {

  @Override
  public ProviderApi getConfigFeatureFlagProviderConstructor() {
    return new FakeProviderApi("FeatureFlagInfo");
  }

  @Override
  public StarlarkExposedRuleTransitionFactory createConfigFeatureFlagTransitionFactory(
      String attribute) {
    return new FakeConfigFeatureFlagTransitionFactory();
  }
}
