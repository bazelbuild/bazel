package build.stack.devtools.build.constellate.rendering;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;

/** Stores information about a Starlark rule definition. */
public class RuleInfoWrapper {

  private final StarlarkCallable identifierFunction;
  private final Location location;
  // Only the Builder is passed to RuleInfoWrapper as the rule name is not yet
  // available.
  private final RuleInfo.Builder ruleInfo;

  public RuleInfoWrapper(
      StarlarkCallable identifierFunction, Location location, RuleInfo.Builder ruleInfo) {
    this.identifierFunction = identifierFunction;
    this.location = location;
    this.ruleInfo = ruleInfo;
  }

  public StarlarkCallable getIdentifierFunction() {
    return identifierFunction;
  }

  public Location getLocation() {
    return location;
  }

  public RuleInfo.Builder getRuleInfo() {
    return ruleInfo;
  }
}
