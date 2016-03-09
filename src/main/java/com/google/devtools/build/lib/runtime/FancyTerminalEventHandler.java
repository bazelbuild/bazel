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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LineCountingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LineWrappingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.OutErr;

import org.joda.time.Duration;
import org.joda.time.Instant;

import java.io.IOException;
import java.util.Calendar;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * An event handler for ANSI terminals which uses control characters to
 * provide eye-candy, reduce scrolling, and generally improve usability
 * for users running directly from the shell.
 *
 * <p/>
 * This event handler differs from a normal terminal because it only adds
 * control characters to stderr, not stdout.  All blaze status feedback
 * is sent to stderr, so adding control characters just to that stream gives
 * the benefits described above without modifying the normal output stream.
 * For commands like build that don't generate stdout output this doesn't
 * matter, but for commands like query and ide_build_info, inserting these
 * control characters in stdout invalidated their output.
 *
 * <p/>
 * The underlying streams may be either line-bufferred or unbuffered.
 * Normally each event will write out a sequence of output to a single
 * stream, and will end with a newline, which ensures a flush.
 * But care is required when outputting incomplete lines, or when mixing
 * output between the two different streams (stdout and stderr):
 * it may be necessary to explicitly flush the output in those cases.
 * However, we also don't want to flush too often; that can lead to
 * a choppy UI experience.
 */
public class FancyTerminalEventHandler extends BlazeCommandEventHandler {
  private static Logger LOG = Logger.getLogger(FancyTerminalEventHandler.class.getName());
  private static final Pattern progressPattern = Pattern.compile(
      // Match strings that look like they start with progress info:
      //   [42%] Compiling base/base.cc
      //   [1,442 / 23,476] Compiling base/base.cc
      "^\\[(?:(?:\\d\\d?\\d?%)|(?:[\\d+,]+ / [\\d,]+))\\] ");
  private static final Splitter LINEBREAK_SPLITTER = Splitter.on('\n');
  private static final List<String> SPECIAL_MESSAGES =
      ImmutableList.of(
          "Reading startup options from "
              + "HKEY_LOCAL_MACHINE\\Software\\Google\\Devtools\\Blaze\\CurrentVersion",
          "Contacting ftp://microsoft.com/win3.1/downloadcenter",
          "Downloading MSVCR71.DLL",
          "Installing Windows Update 37 of 118...",
          "Sending request to Azure server",
          "Checking whether your copy of Blaze is Genuine",
          "Initializing HAL",
          "Loading NDIS2SUP.VXD",
          "Initializing DRM",
          "Contacting license server",
          "Starting EC2 instances",
          "Starting MS-DOS 6.0",
          "Updating virus database",
          "Linking WIN32.DLL",
          "Linking GGL32.EXE",
          "Starting ActiveX controls",
          "Launching Microsoft Visual Studio 2013",
          "Launching IEXPLORE.EXE",
          "Initializing BASIC v2.1 interpreter",
          "Parsing COM object monikers",
          "Notifying field agents",
          "Negotiating with killer robots",
          "Searching for cellular signal",
          "Checking for outstanding GCard expenses",
          "Waiting for workstation CPU temperature to decrease");

  private static final Set<Character> PUNCTUATION_CHARACTERS =
      ImmutableSet.<Character>of(',', '.', ':', '?', '!', ';');

  private final Iterator<String> messageIterator = Iterators.cycle(SPECIAL_MESSAGES);
  private volatile boolean trySpecial;
  private volatile Instant skipUntil = Instant.now();

  private final AnsiTerminal terminal;

  private final boolean useColor;
  private final boolean useCursorControls;
  private final boolean progressInTermTitle;
  public final int terminalWidth;

  private boolean terminalClosed = false;
  private boolean previousLineErasable = false;
  private int numLinesPreviousErasable = 0;

  public FancyTerminalEventHandler(OutErr outErr, BlazeCommandEventHandler.Options options) {
    super(outErr, options);
    this.terminal = new AnsiTerminal(outErr.getErrorStream());
    this.terminalWidth = (options.terminalColumns > 0 ? options.terminalColumns : 80);
    useColor = options.useColor();
    useCursorControls = options.useCursorControl();
    progressInTermTitle = options.progressInTermTitle;
    
    Calendar today = Calendar.getInstance();
    trySpecial = options.forceExternalRepositories 
        || (options.externalRepositories
            && today.get(Calendar.MONTH) == Calendar.APRIL
            && today.get(Calendar.DAY_OF_MONTH) == 1);
  }

  @Override
  public void handle(Event event) {
    if (terminalClosed) {
      return;
    }
    if (!eventMask.contains(event.getKind())) {
      return;
    }
    if (trySpecial && !EventKind.ERRORS_AND_WARNINGS_AND_OUTPUT.contains(event.getKind())
        && skipUntil.isAfterNow()) {
      // Short-circuit here to avoid wiping out previous terminal contents.
      return;
    }

    try {
      boolean previousLineErased = false;
      if (previousLineErasable) {
        previousLineErased = maybeOverwritePreviousMessage();
      }
      switch (event.getKind()) {
        case PROGRESS:
        case START:
          {
            String message = event.getMessage();
            Pair<String,String> progressPair = matchProgress(message);
            if (progressPair != null) {
              progress(progressPair.getFirst(), progressPair.getSecond());
              if (trySpecial && ThreadLocalRandom.current().nextInt(0, 20) == 0) {
                message = getExtraMessage();
                if (message != null) {
                  // Should always be true, but don't crash on that!
                  previousLineErased = maybeOverwritePreviousMessage();
                  progress(progressPair.getFirst(), message);
                  // Skip unimportant messages for a bit so that this message gets some exposure.
                  skipUntil = Instant.now().plus(
                      Duration.millis(ThreadLocalRandom.current().nextInt(3000, 8000)));
                }
              }
            } else {
              progress("INFO: ", message);
            }
            break;
          }
        case FINISH:
          {
            String message = event.getMessage();
            Pair<String,String> progressPair = matchProgress(message);
            if (progressPair != null) {
              String percentage = progressPair.getFirst();
              String rest = progressPair.getSecond();
              progress(percentage, rest + " DONE");
            } else {
              progress("INFO: ", message + " DONE");
            }
            break;
          }
        case PASS:
          progress("PASS: ", event.getMessage());
          break;
        case INFO:
          info(event);
          break;
        case ERROR:
        case FAIL:
        case TIMEOUT:
          // For errors, scroll the message, so it appears above the status
          // line, and highlight the word "ERROR" or "FAIL" in boldface red.
          errorOrFail(event);
          break;
        case WARNING:
          // For warnings, highlight the word "Warning" in boldface magenta,
          // and scroll it.
          warning(event);
          break;
        case SUBCOMMAND:
          subcmd(event);
          break;
        case STDOUT:
          if (previousLineErased) {
            terminal.flush();
          }
          previousLineErasable = false;
          super.handle(event);
          // We don't need to flush stdout here, because
          // super.handle(event) will take care of that.
          break;
        case STDERR:
          putOutput(event);
          break;
        default:
          // Ignore all other event types.
          break;
      }
    } catch (IOException e) {
      // The terminal shouldn't have IO errors, unless the shell is killed, which
      // should also kill the blaze client. So this isn't something that should
      // occur here; it will show up in the client/server interface as a broken
      // pipe.
      LOG.warning("Terminal was closed during build: " + e);
      terminalClosed = true;
    }
  }
  
  private String getExtraMessage() {
    synchronized (messageIterator) {
      if (messageIterator.hasNext()) {
        return messageIterator.next();
      }
    }
    trySpecial = false;
    return null;
  }

  /**
   * Displays a progress message that may be erased by subsequent messages.
   *
   * @param  prefix   a short string such as "[99%] " or "INFO: ", which will be highlighted
   * @param  rest     the remainder of the message; may be multiple lines
   */
  private void progress(String prefix, String rest) throws IOException {
    previousLineErasable = true;

    if (progressInTermTitle) {
      int newlinePos = rest.indexOf('\n');
      if (newlinePos == -1) {
        terminal.setTitle(prefix + rest);
      } else {
        terminal.setTitle(prefix + rest.substring(0, newlinePos));
      }
    }

    LineCountingAnsiTerminalWriter countingWriter = new LineCountingAnsiTerminalWriter(terminal);
    AnsiTerminalWriter terminalWriter =
        new LineWrappingAnsiTerminalWriter(countingWriter, terminalWidth - 1);

    if (useColor) {
      terminalWriter.okStatus();
    }
    terminalWriter.append(prefix);
    terminalWriter.normal();
    if (showTimestamp) {
      String timestamp = timestamp();
      terminalWriter.append(timestamp);
    }
    Iterator<String> lines = LINEBREAK_SPLITTER.split(rest).iterator();
    String firstLine = lines.next();
    terminalWriter.append(firstLine);
    terminalWriter.newline();
    while (lines.hasNext()) {
      String line = lines.next();
      terminalWriter.append(line);
      terminalWriter.newline();
    }
    numLinesPreviousErasable = countingWriter.getWrittenLines();
  }

  /**
   * Try to match a message against the "progress message" pattern. If it
   * matches, return the progress percentage, and the rest of the message.
   * @param message the message to match
   * @return a pair containing the progress percentage, and the rest of the
   *    progress message, or null if the message isn't a progress message.
   */
  private Pair<String,String> matchProgress(String message) {
    Matcher m = progressPattern.matcher(message);
    if (m.find()) {
      return Pair.of(message.substring(0, m.end()), message.substring(m.end()));
    } else {
      return null;
    }
  }

  /**
   * Send the terminal controls that will put the cursor on the beginning
   * of the same line if cursor control is on, or the next line if not.
   * @return True if it did any output; if so, caller is responsible for
   *          flushing the terminal if needed.
   */
  private boolean maybeOverwritePreviousMessage() throws IOException {
    if (useCursorControls && numLinesPreviousErasable != 0) {
      for (int i = 0; i < numLinesPreviousErasable; i++) {
        terminal.cr();
        terminal.cursorUp(1);
        terminal.clearLine();
      }
      return true;
    } else {
      return false;
    }
  }

  private void errorOrFail(Event event) throws IOException {
    previousLineErasable = false;
    if (useColor) {
      terminal.textRed();
      terminal.textBold();
    }
    terminal.writeString(event.getKind() + ": ");
    if (useColor) {
      terminal.resetTerminal();
    }
    writeTimestampAndLocation(event);
    writeStringWithPotentialPeriod(event.getMessage());
    crlf();
  }

  private void warning(Event warning) throws IOException {
    previousLineErasable = false;
    if (useColor) {
      terminal.textMagenta();
    }
    terminal.writeString("WARNING: ");
    terminal.resetTerminal();
    writeTimestampAndLocation(warning);
    writeStringWithPotentialPeriod(warning.getMessage());
    crlf();
  }

  private void info(Event event) throws IOException {
    previousLineErasable = false;
    if (useColor) {
      terminal.textGreen();
    }
    terminal.writeString(event.getKind() + ": ");
    terminal.resetTerminal();
    writeTimestampAndLocation(event);
    terminal.writeString(event.getMessage());
    // No period; info messages may end with a URL.
    crlf();
  }

  /**
   * Writes the given String to the terminal. This method also writes a trailing period if the
   * message doesn't end with a punctuation character.
   */
  private void writeStringWithPotentialPeriod(String message) throws IOException {
    terminal.writeString(message);
    if (!message.isEmpty()) {
      char lastChar = message.charAt(message.length() - 1);
      if (!PUNCTUATION_CHARACTERS.contains(lastChar)) {
        terminal.writeString(".");
      }
    }
  }

  private void subcmd(Event subcmd) throws IOException {
    previousLineErasable = false;
    if (useColor) {
      terminal.textBlue();
    }
    terminal.writeString(">>>>> ");
    terminal.resetTerminal();
    writeTimestampAndLocation(subcmd);
    terminal.writeString(subcmd.getMessage());
    crlf();
  }

  /* Handle STDERR events. */
  private void putOutput(Event event) throws IOException {
    previousLineErasable = false;
    terminal.writeBytes(event.getMessageBytes());
/*
 * The following code doesn't work because buildtool.TerminalTestNotifier
 * writes ANSI-formatted text via this mechanism, one character at a time,
 * and if we try to insert additional ANSI sequences in between the characters
 * of another ANSI escape sequence, we screw things up. (?)
 * TODO(bazel-team): (2009) fix this.  TerminalTestNotifier should go via the Reporter
 * rather than via an AnsiTerminalWriter.
 */
//    terminal.resetTerminal();
//    writeTimestampAndLocation(event);
//    if (useColor) {
//      terminal.textNormal();
//    }
//    terminal.writeBytes(event.getMessageBytes());
//    terminal.resetTerminal();
  }

  /**
   * Add a carriage return, shifting to the next line on the terminal, while
   * guaranteeing that the terminal control codes don't cause any strange
   * effects.  Without the CR before the "\n", the "\n" can cause a line-break
   * moving text to the next line, where the new message will be generated.
   * Emitting a "CR" before means that the actual terminal controls generated
   * here are CR+CR+LF; the double-CR resets the terminal line state, which
   * prevents the potentially ugly formatting issue.
   */
  private void crlf() throws IOException {
    terminal.cr();
    terminal.writeString("\n");
  }

  private void writeTimestampAndLocation(Event event) throws IOException {
    if (showTimestamp) {
      terminal.writeString(timestamp());
    }
    if (event.getLocation() != null) {
      terminal.writeString(event.getLocation() + ": ");
    }
  }

  public void resetTerminal() {
    try {
      terminal.resetTerminal();
    } catch (IOException e) {
      LOG.warning("IO Error writing to user terminal: " + e);
    }
  }
}
