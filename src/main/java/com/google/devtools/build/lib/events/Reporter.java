// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.events;

import com.google.devtools.build.lib.util.io.OutErr;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * The reporter is the primary means of reporting events such as errors,
 * warnings, progress information and diagnostic information to the user.  It
 * is not intended as a logging mechanism for developer-only messages; use a
 * Logger for that.
 *
 * The reporter instance is consumed by the build system, and passes events to
 * {@link EventHandler} instances. These handlers are registered via {@link
 * #addHandler(EventHandler)}.
 *
 * <p>Thread-safe: calls to {@code #report} may be made on any thread.
 * Handlers may be run in an arbitary thread (but right now, they will not be
 * run concurrently).
 */
public final class Reporter implements ErrorEventListener, ExceptionListener {

  private final List<EventHandler> handlers = new ArrayList<>();

  /** The set of event kinds that we currently have a handler for. */
  private final Set<EventKind> mask = EnumSet.noneOf(EventKind.class);

  /** An OutErr that sends all of its output to this Reporter.
   * Each write will (when flushed) get mapped to an EventKind.STDOUT or EventKind.STDERR event.
   */
  private final OutErr outErrToReporter = OutErr.create(
      // We don't use BufferedOutputStream here, because in general the Blaze
      // code base assumes that the output streams are not buffered.
      new ReporterStream(this, EventKind.STDOUT),
      new ReporterStream(this, EventKind.STDERR));

  private volatile OutputFilter outputFilter = OutputFilter.OUTPUT_EVERYTHING;

  public Reporter() {}

  /**
   * A copy constructor, to make it convenient to replicate a reporter
   * config for temporary configuration changes.
   */
  public Reporter(Reporter template) {
    handlers.addAll(template.handlers);
    recomputeMask();
  }

  /**
   * Constructor which configures a reporter with the specified handlers.
   */
  public Reporter(EventHandler... handlers) {
    for (EventHandler handler: handlers) {
      addHandler(handler);
    }
    recomputeMask();
  }

  private void recomputeMask() {
    // mask = union of hander.getEventMask()
    mask.clear();
    for (EventHandler handler: handlers) {
      mask.addAll(handler.getEventMask());
    }
  }

  /**
   * Returns an OutErr that sends all of its output to this Reporter.
   * Each write to the OutErr will cause an EventKind.STDOUT or EventKind.STDERR event.
   */
  public OutErr getOutErr() {
    return outErrToReporter;
  }

  /**
   * Adds a handler to this reporter.
   */
  public synchronized void addHandler(EventHandler handler) {
    handlers.add(handler);
    mask.addAll(handler.getEventMask());
  }

  /**
   * Removes handler from this reporter.
   */
  public synchronized void removeHandler(EventHandler handler) {
    handlers.remove(handler);
    recomputeMask();
  }

  /**
   * Returns true iff this Reporter has an EventHandler that will handle the
   * specified event kind.  Callers may use this to optimise away unnecessary
   * calls to the Reporter when no-one is listening.
   */
  public boolean hasHandlerFor(EventKind kind) {
    return mask.contains(kind);
  }

  /**
   * This method is called by the build system to report an event.
   */
  @Override
  public synchronized void report(EventKind kind, Location location,
                                   String message) {
    Event event = null; // Created lazily iff required.
    if (hasHandlerFor(kind)) {
      for (EventHandler handler : handlers) {
        if (handler.getEventMask().contains(kind)) {
          if (event == null) {
            event = new Event(kind, location, message);
          }
          handler.handle(event);
        }
      }
    }
  }

  /**
   * Variant of report() for a byte-array.
   */
  public synchronized void report(EventKind kind, Location location,
                                   byte[] messageBytes) {
    Event event = null; // Created lazily iff required.
    if (hasHandlerFor(kind)) {
      for (EventHandler handler : handlers) {
        if (handler.getEventMask().contains(kind)) {
          if (event == null) {
            event = new Event(kind, location, messageBytes);
          }
          handler.handle(event);
        }
      }
    }
  }

  /**
   * Reports a subcommand executed by the build tool; a wrapper around
   * report() with event kind SUBCOMMAND.
   */
  public void subcommand(Location location, String message) {
    report(EventKind.SUBCOMMAND, location, message);
  }

  /**
   * Reports atemporal statements about the build, i.e. they're true for the
   * duration of execution.  (e.g. how many targets were found.)
   * Is a wrapper around report() with event kind INFO.
   * This information is always displayed.
   */
  @Override
  public void info(Location location, String message) {
    report(EventKind.INFO, location, message);
  }

  /**
   * Reports temporal statements about how far the build has progressed
   * and what it is currently doing.  (e.g. what's executing now.)
   * Is a wrapper around report() with event kind PROGRESS.
   */
  @Override
  public void progress(Location location, String message) {
    report(EventKind.PROGRESS, location, message);
  }

  /**
   * Reports the start of a particular task.
   * Is a wrapper around report() with event kind START.
   * Should always be matched by a corresponding call to finishTask()
   * with the same message, except that the leading percentage
   * progress indicator (if any) in the message may differ.
   */
  public void startTask(Location location, String message) {
    report(EventKind.START, location, message);
  }

  /**
   * Reports the start of a particular task.
   * Is a wrapper around report() with event kind FINISH.
   * Should always be matched by a corresponding call to startTask()
   * with the same message, except that the leading percentage
   * progress indicator (if any) in the message may differ.
   */
  public void finishTask(Location location, String message) {
    report(EventKind.FINISH, location, message);
  }

  @Override
  public void warn(Location location, String message) {
    report(EventKind.WARNING, location, message);
  }

  @Override
  public void error(Location location, String message) {
    report(EventKind.ERROR, location, message);
  }

  @Override
  public void error(Location location, String message, Throwable error) {
    error(location, message);
    error.printStackTrace(new PrintStream(getOutErr().getErrorStream()));
  }

  /**
   * Reports a dependency-checking event, which explains why an Action is being
   * rebuilt (or not).
   */
  public void depchecker(String message) {
    report(EventKind.DEPCHECKER, null, message);
  }

  /**
   * Returns true iff the given tag matches the output filter.
   */
  @Override
  public boolean showOutput(String tag) {
    return outputFilter.showOutput(tag);
  }

  public void setOutputFilter(OutputFilter outputFilter) {
    this.outputFilter = outputFilter;
  }
}
