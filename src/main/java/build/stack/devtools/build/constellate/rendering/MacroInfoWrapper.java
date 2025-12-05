package build.stack.devtools.build.constellate.rendering;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;

/** Stores information about a Starlark symbolic macro definition. */
public class MacroInfoWrapper {

  private final StarlarkCallable identifierFunction;
  private final Location location;
  // Only the Builder is passed to MacroInfoWrapper as the macro name is not yet
  // available.
  private final MacroInfo.Builder macroInfo;

  public MacroInfoWrapper(
      StarlarkCallable identifierFunction, Location location, MacroInfo.Builder macroInfo) {
    this.identifierFunction = identifierFunction;
    this.location = location;
    this.macroInfo = macroInfo;
  }

  public StarlarkCallable getIdentifierFunction() {
    return identifierFunction;
  }

  public Location getLocation() {
    return location;
  }

  public MacroInfo.Builder getMacroInfo() {
    return macroInfo;
  }
}
