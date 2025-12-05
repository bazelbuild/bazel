package build.stack.devtools.build.constellate;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

/**
 * A simple {@link EventHandler} which outputs log information to system.out and system.err.
 */
class SystemOutEventHandler implements EventHandler {

  @Override
  public void handle(Event event) {
    switch (event.getKind()) {
      case ERROR:
      case WARNING:
      case STDERR:
        System.err.println(messageWithLocation(event));
        break;
      case DEBUG:
      case INFO:
      case PROGRESS:
      case STDOUT:
        System.out.println(messageWithLocation(event));
        break;
      default:
        System.err.println("Unknown message type: " + event);
    }
  }

  private String messageWithLocation(Event event) {
    String location =
        event.getLocation() == null ? "<no location>" : event.getLocation().toString();
    return location + ": " + event.getMessage();
  }
}
