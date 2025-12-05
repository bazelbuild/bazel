package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Fake callable implementation of {@link ProviderApi}. */
public class FakeProviderApi implements StarlarkCallable, ProviderApi {

  private String name;

  public FakeProviderApi(@Nullable String name) {
    this.name = name;
  }

  @Override
  public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
    return new FakeStructApi();
  }

  @Override
  public String getName() {
    return name != null ? name : "Unexported Provider";
  }

  /** Called when provider is "exported" by a top-level assignment {@code name = provider()}. */
  public void setName(String name) {
    if (this.name == null) {
      this.name = name;
    }
  }

  @Override
  public void repr(Printer printer) {}
}
