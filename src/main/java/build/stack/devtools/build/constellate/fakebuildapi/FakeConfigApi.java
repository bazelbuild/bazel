package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi.BuildSettingApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi.ExecTransitionFactoryApi;
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
  public BuildSettingApi stringListSetting(Boolean flag, Boolean repeatable) {
    return new FakeBuildSettingDescriptor();
  }

  @Override
  public BuildSettingApi stringSetSetting(Boolean flag, Boolean repeatable) {
    return new FakeBuildSettingDescriptor();
  }

  @Override
  public ExecTransitionFactoryApi exec(Object execGroup) {
    return new FakeExecTransitionFactory();
  }

  @Override
  public ExecTransitionFactoryApi target() {
    return new FakeExecTransitionFactory();
  }

  @Override
  public ExecTransitionFactoryApi none() {
    return new FakeExecTransitionFactory();
  }

  @Override
  public void repr(Printer printer) {}
}
