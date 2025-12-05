package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAspectApi;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkCallable;

/** Fake implementation of {@link StarlarkAspectApi}. */
public class FakeStarlarkAspect implements StarlarkCallable, StarlarkAspectApi {

  /**
   * Each fake is constructed with a unique name, controlled by this counter being the name suffix.
   */
  private static int idCounter = 0;

  private final String name = "AspectIdentifier" + idCounter++;

  @Override
  public String getName() {
    return name;
  }

  @Override
  public void repr(Printer printer) {}
}
