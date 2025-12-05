package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import net.starlark.java.eval.Printer;

/** Fake implementation of {@link SplitTransitionProviderApi}. */
public class FakeSplitTransitionProvider implements SplitTransitionProviderApi {

  @Override
  public void repr(Printer printer) {}
}
