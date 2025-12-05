package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.LateBoundDefaultApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkBuildApiGlobals;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** Fake implementation of {@link StarlarkBuildApiGlobals}. */
public class FakeBuildApiGlobals implements StarlarkBuildApiGlobals {

  @Override
  public LateBoundDefaultApi configurationField(String fragment, String name, StarlarkThread thread)
      throws EvalException {
    return new FakeLateBoundDefaultApi();
  }
}
