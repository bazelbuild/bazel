package net.starlark.java.eval;

import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.syntax.Location;

/**
 * A {@link StarlarkValue} that can be used as a cause in calls to {@code link}.
 *
 * <p>Implementations should override {@link StarlarkValue#debugPrint(Printer, StarlarkSemantics)}
 * and ensure that, combined with {@link FailureCause#getLocation()}, this provides sufficient
 * information for users to understand the cause of a failure without a stack trace.
 */
@StarlarkBuiltin(name = "failure_cause",
    doc = "Can be passed to the <code>cause</code> parameter of "
        + "<a href=\"globals.html#fail\"><code>fail</code></a> to add context to an error message.")
public interface FailureCause extends StarlarkValue {

  Location getLocation();

  default void appendCauseTo(Printer printer, StarlarkSemantics semantics) {
    printer.append(Starlark.type(this));
    Location location = getLocation();
    printer.append(" in file \"");
    printer.append(location.file());
    printer.append('\"');
    if (location.line() != 0) {
      printer.append(", line ");
      printer.append(location.line());
      if (location.column() != 0) {
        printer.append(", column ");
        printer.append(location.column());
      }
    }
    printer.append(", with value:\n");
    try (SilentCloseable ignored = printer.indent()) {
      debugPrint(printer, semantics);
    }
  }
}
