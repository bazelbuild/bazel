package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.LateBoundDefaultApi;
import net.starlark.java.eval.Printer;

/**
 * Fake implementation of {@link LateBoundDefaultApi}.
 */
public class FakeLateBoundDefaultApi implements LateBoundDefaultApi {

  @Override
  public void repr(Printer printer) {}
}
