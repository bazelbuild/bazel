package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi.BuildSettingApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi.ExecTransitionFactoryApi;
import net.starlark.java.eval.Printer;

/** Fakes for callables under the {@link StarlarkConfigApi} module. */
public class ConfigApiFakes {

  private ConfigApiFakes() {}

  /** Fake implementation of {@link BuildSettingApi}. */
  public static class FakeBuildSettingDescriptor implements BuildSettingApi {

    @Override
    public void repr(Printer printer) {}
  }

  /** Fake implementation of ExecTransitionFactoryApi. */
  public static class FakeExecTransitionFactory implements ExecTransitionFactoryApi {
    @Override
    public void repr(Printer printer) {}
  }
}
