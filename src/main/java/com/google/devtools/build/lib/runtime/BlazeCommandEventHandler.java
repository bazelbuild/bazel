// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.EnumSet;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * BlazeCommandEventHandler: an event handler established for the duration of a
 * single Blaze command.
 */
public class BlazeCommandEventHandler implements EventHandler {

  private static final Logger LOG = Logger.getLogger(BlazeCommandEventHandler.class.getName());

  public enum UseColor { YES, NO, AUTO }
  public enum UseCurses { YES, NO, AUTO }

  public static class UseColorConverter extends EnumConverter<UseColor> {
    public UseColorConverter() {
      super(UseColor.class, "--color setting");
    }
  }

  public static class UseCursesConverter extends EnumConverter<UseCurses> {
    public UseCursesConverter() {
      super(UseCurses.class, "--curses setting");
    }
  }

  public static class Options extends OptionsBase {

    @Option(name = "show_progress",
            defaultValue = "true",
            category = "verbosity",
            help = "Display progress messages during a build.")
    public boolean showProgress;

    @Option(name = "show_task_finish",
            defaultValue = "false",
            category = "verbosity",
            help = "Display progress messages when tasks complete, not just when they start.")
    public boolean showTaskFinish;

    @Option(name = "show_progress_rate_limit",
            defaultValue = "0.03",  // A nice middle ground; snappy but not too spammy in logs.
            category = "verbosity",
            help = "Minimum number of seconds between progress messages in the output.")
    public double showProgressRateLimit;

    @Option(name = "color",
            defaultValue = "auto",
            converter = UseColorConverter.class,
            category = "verbosity",
            help = "Use terminal controls to colorize output.")
    public UseColor useColorEnum;

    @Option(name = "curses",
            defaultValue = "auto",
            converter = UseCursesConverter.class,
            category = "verbosity",
            help = "Use terminal cursor controls to minimize scrolling output")
    public UseCurses useCursesEnum;

    @Option(name = "terminal_columns",
            defaultValue = "80",
            category = "hidden",
            help = "A system-generated parameter which specifies the terminal "
               + " width in columns.")
    public int terminalColumns;

    @Option(name = "isatty",
            defaultValue = "false",
            category = "hidden",
            help = "A system-generated parameter which is used to notify the "
                + "server whether this client is running in a terminal. "
                + "If this is set to false, then '--color=auto' will be treated as '--color=no'. "
                + "If this is set to true, then '--color=auto' will be treated as '--color=yes'.")
    public boolean isATty;

    // This lives here (as opposed to the more logical BuildRequest.Options)
    // because the client passes it to the server *always*.  We don't want the
    // client to have to figure out when it should or shouldn't to send it.
    @Option(name = "emacs",
            defaultValue = "false",
            category = "undocumented",
            help = "A system-generated parameter which is true iff EMACS=t in the environment of "
               + "the client.  This option controls certain display features.")
    public boolean runningInEmacs;

    @Option(name = "show_timestamps",
        defaultValue = "false",
        category = "verbosity",
        help = "Include timestamps in messages")
    public boolean showTimestamp;

    @Option(name = "progress_in_terminal_title",
        defaultValue = "false",
        category = "verbosity",
        help = "Show the command progress in the terminal title. "
            + "Useful to see what blaze is doing when having multiple terminal tabs.")
    public boolean progressInTermTitle;

    @Option(name = "experimental_external_repositories",
        defaultValue = "false",
        category = "verbosity",
        help = "Use external repositories for improved stability and speed when available.")
    public boolean externalRepositories;

    @Option(name = "force_experimental_external_repositories",
        defaultValue = "false",
        category = "verbosity",
        help = "Forces --experimental_external_repositories.")
    public boolean forceExternalRepositories;

    @Option(
      name = "experimental_ui",
      defaultValue = "false",
      category = "hidden",
      help = "Enable the experimental new Bazel UI."
    )
    public boolean experimentalUi;

    public boolean useColor() {
      return useColorEnum == UseColor.YES || (useColorEnum == UseColor.AUTO && isATty);
    }

    public boolean useCursorControl() {
      return useCursesEnum == UseCurses.YES || (useCursesEnum == UseCurses.AUTO && isATty);
    }
  }

  private static final DateTimeFormatter TIMESTAMP_FORMAT =
      DateTimeFormat.forPattern("(MM-dd HH:mm:ss.SSS) ");

  protected final OutErr outErr;

  private final PrintStream errPrintStream;

  protected final Set<EventKind> eventMask =
      EnumSet.copyOf(EventKind.ERRORS_WARNINGS_AND_INFO_AND_OUTPUT);

  protected final boolean showTimestamp;

  public BlazeCommandEventHandler(OutErr outErr, Options eventOptions) {
    this.outErr = outErr;
    this.errPrintStream = new PrintStream(outErr.getErrorStream(), true);
    if (eventOptions.showProgress) {
      eventMask.add(EventKind.PROGRESS);
      eventMask.add(EventKind.START);
    } else {
      // Skip PASS events if --noshow_progress is requested.
      eventMask.remove(EventKind.PASS);
    }
    if (eventOptions.showTaskFinish) {
      eventMask.add(EventKind.FINISH);
    }
    eventMask.add(EventKind.SUBCOMMAND);
    this.showTimestamp = eventOptions.showTimestamp;
  }

  /** See EventHandler.handle. */
  @Override
  public void handle(Event event) {
    if (!eventMask.contains(event.getKind())) {
      return;
    }
    String prefix;
    switch (event.getKind()) {
      case STDOUT:
        putOutput(outErr.getOutputStream(), event);
        return;
      case STDERR:
        putOutput(outErr.getErrorStream(), event);
        return;
      case PASS:
      case FAIL:
      case TIMEOUT:
      case ERROR:
      case WARNING:
      case DEPCHECKER:
        prefix = event.getKind() + ": ";
        break;
      case SUBCOMMAND:
        prefix = ">>>>>>>>> ";
        break;
      case INFO:
      case PROGRESS:
      case START:
      case FINISH:
        prefix = "____";
        break;
      default:
        throw new IllegalStateException("" + event.getKind());
    }
    StringBuilder buf = new StringBuilder();
    buf.append(prefix);

    if (showTimestamp) {
      buf.append(timestamp());
    }

    Location location = event.getLocation();
    if (location != null) {
      buf.append(location.print()).append(": ");
    }

    buf.append(event.getMessage());
    if (event.getKind() == EventKind.FINISH) {
      buf.append(" DONE");
    }

    // Add a trailing period for ERROR and WARNING messages, which are
    // typically English sentences composed from exception messages.
    if (event.getKind() == EventKind.WARNING ||
        event.getKind() == EventKind.ERROR) {
      buf.append('.');
    }

    // Event messages go to stderr; results (e.g. 'blaze query') go to stdout.
    errPrintStream.println(buf);
  }

  private void putOutput(OutputStream out, Event event) {
    try {
      out.write(event.getMessageBytes());
      out.flush();
    } catch (IOException e) {
      // This can happen in server mode if the blaze client has exited, or if output is redirected
      // to a file and the disk is full, etc. May be moot in the case of full disk, or useful in
      // the case of real bug in our handling of streams.
      LOG.log(Level.WARNING, "Failed to write event", e);
    }
  }

  /**
   * @return a string representing the current time, eg "04-26 13:47:32.124".
   */
  protected String timestamp() {
    return TIMESTAMP_FORMAT.print(System.currentTimeMillis());
  }
}
