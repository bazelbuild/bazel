package com.google.devtools.build.lib.events;

import com.google.common.flogger.GoogleLogger;

import java.util.logging.Level;

/**
 * FixedLevelLoggingEventHandler logs events to a GoogleLogger at a fixed level.
 */
public class FixedLevelLoggingEventHandler implements EventHandler {
    private final GoogleLogger logger;
    private final Level logLevel;

    public FixedLevelLoggingEventHandler(GoogleLogger logger, Level logLevel) {
        this.logger = logger;
        this.logLevel = logLevel;
    }

    @Override
    public void handle(Event event) {
        logger.at(logLevel).log("%s", event.getMessage());
    }
}
