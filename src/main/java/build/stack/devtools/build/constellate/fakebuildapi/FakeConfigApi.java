package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.StarlarkConfigApi;
import build.stack.devtools.build.constellate.fakebuildapi.ConfigApiFakes.FakeBuildSettingDescriptor;
import build.stack.devtools.build.constellate.fakebuildapi.ConfigApiFakes.FakeExecTransitionFactory;
import net.starlark.java.eval.Printer;

/** Fake implementation of {@link StarlarkConfigApi}. */
public class FakeConfigApi implements StarlarkConfigApi {

  @Override
  public BuildSettingApi intSetting(Boolean flag) {
    return new FakeBuildSettingDescriptor();
  }

  @Override
  public BuildSettingApi boolSetting(Boolean flag) {
    return new FakeBuildSettingDescriptor();
  }

  @Override
  public BuildSettingApi stringSetting(Boolean flag, Boolean allowMultiple) {
    return new FakeBuildSettingDescriptor();
  }

  @Override
  public BuildSettingApi stringListSetting(Boolean flag) {
    return new FakeBuildSettingDescriptor();
  }

  @Override
  public ExecTransitionFactoryApi exec(Object execGroup) {
    return new FakeExecTransitionFactory();
  }

  @Override
  public void repr(Printer printer) {}
}
