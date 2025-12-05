package build.stack.devtools.build.constellate.fakebuildapi.config;

import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigStarlarkCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import build.stack.devtools.build.constellate.fakebuildapi.FakeDeepStructure;
import build.stack.devtools.build.constellate.fakebuildapi.FakeProviderApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Structure;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import javax.annotation.Nullable;

/**
 * Fake implementation of {@link ConfigStarlarkCommonApi}.
 * Implements Structure to support dynamic attribute access for methods not in the interface.
 */
public class FakeConfigStarlarkCommon implements ConfigStarlarkCommonApi, Structure {

  private final FakeDeepStructure delegate = FakeDeepStructure.create("config_common");

  @Override
  public ProviderApi getConfigFeatureFlagProviderConstructor() {
    return new FakeProviderApi("FeatureFlagInfo");
  }

  @Override
  public StarlarkExposedRuleTransitionFactory createConfigFeatureFlagTransitionFactory(
      String attribute) {
    return new FakeConfigFeatureFlagTransitionFactory();
  }

  // Delegate to FakeDeepStructure for dynamic attribute access
  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    return delegate.getValue(name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ImmutableList.of("FeatureFlagInfo", "config_feature_flag_transition");
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return null; // Return null to allow dynamic field access
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<config_common>");
  }
}
