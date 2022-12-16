package com.google.devtools.build.lib.events;

/** TeeEventHandler forwards events to two delegate EventHandlers. */
public class TeeEventHandler implements EventHandler {
    private final EventHandler delegate1;
    private final EventHandler delegate2;

    public TeeEventHandler(EventHandler delegate1, EventHandler delegate2) {
        this.delegate1 = delegate1;
        this.delegate2 = delegate2;
    }

    @Override
    public void handle(Event event) {
        delegate1.handle(event);
        delegate2.handle(event);
    }
}
