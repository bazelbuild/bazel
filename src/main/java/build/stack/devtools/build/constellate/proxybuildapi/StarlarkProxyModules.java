package build.stack.devtools.build.constellate.proxybuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.packages.StarlarkLibrary;
import com.google.devtools.build.lib.packages.StructProvider;
import net.starlark.java.eval.Starlark;

/** A helper class for determining essential Build Language builtins. */
public final class StarlarkModules {

  private StarlarkModules() {}

  /**
   * Adds essential predeclared symbols for the Build Language.
   *
   * <p>This includes generic symbols like {@code rule()}, but not symbols specific to a rule
   * family, like {@code CcInfo}; those are registered on a RuleClassProvider instead. This also
   * does not include Starlark Universe symbols like {@code len()}.
   */
  public static void addPredeclared(ImmutableMap.Builder<String, Object> predeclared) {
    predeclared.putAll(StarlarkLibrary.COMMON); // e.g. select, depset
    Starlark.addMethods(predeclared, new BazelBuildApiGlobals()); // e.g. configuration_field
    Starlark.addMethods(predeclared, ProxyStarlarkRuleFunctionsApi.of(new StarlarkRuleClassFunctions())); // e.g. rule
    predeclared.put("cmd_helper", new StarlarkCommandLine());
    predeclared.put("attr", new StarlarkAttrModule());
    predeclared.put("struct", StructProvider.STRUCT);
    predeclared.put("OutputGroupInfo", OutputGroupInfo.STARLARK_CONSTRUCTOR);
    predeclared.put("Actions", ActionsProvider.INSTANCE);
    predeclared.put("DefaultInfo", DefaultInfo.PROVIDER);
  }
}
