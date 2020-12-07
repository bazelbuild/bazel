// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.util.SimpleLogHandler.HandlerQuerier;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.Date;
import java.util.Optional;
import java.util.TimeZone;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the {@link SimpleLogHandler} class. */
@RunWith(JUnit4.class)
public final class SimpleLogHandlerTest {

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Test
  public void testPrefix() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello_world_%u%h%%_")
            .build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.
    assertThat(handler.getCurrentLogFilePath().get().toString())
        .startsWith(tmp.getRoot() + File.separator + "hello_world_%u%h%%_");
  }

  @Test
  public void testPatternVariables() throws Exception {
    String username = System.getProperty("user.name");
    if (Strings.isNullOrEmpty(username)) {
      username = "unknown_user";
    }
    String hostname = SimpleLogHandler.getLocalHostnameFirstComponent();

    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello_")
            .setPattern("world_%u%%%h_")
            .build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.
    assertThat(handler.getCurrentLogFilePath().get().toString())
        .startsWith(
            tmp.getRoot() + File.separator + "hello_world_" + username + "%" + hostname + "_");
  }

  @Test
  public void testPatternInvalidVariable() throws Exception {
    assertThrows(
        IllegalArgumentException.class,
        () -> SimpleLogHandler.builder().setPattern("hello_%t").build());
  }

  @Test
  public void testExtensionDefaults() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder().setPrefix(tmp.getRoot() + File.separator + "hello").build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.
    assertThat(handler.getCurrentLogFilePath().get().toString())
        .endsWith("." + SimpleLogHandler.getPidString());
  }

  @Test
  public void testExtensionSetter() throws Exception {
    SimpleLogHandler handler1 =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setExtension("xyz")
            .build();
    handler1.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.
    assertThat(handler1.getCurrentLogFilePath().get().toString()).endsWith(".xyz");
  }

  private static final class FakeClock extends Clock {
    private Instant now;
    private final ZoneId zone;

    FakeClock(Instant now, ZoneId zone) {
      this.now = now;
      this.zone = zone;
    }

    void set(Instant now) {
      this.now = now;
    }

    @Override
    public Instant instant() {
      return now;
    }

    @Override
    public ZoneId getZone() {
      return zone;
    }

    @Override
    public Clock withZone(ZoneId zone) {
      return new FakeClock(this.now, zone);
    }
  }

  @Test
  public void testTimestamp() throws Exception {
    Instant instant = Instant.parse("2015-09-01T15:17:54Z");
    FakeClock clock = new FakeClock(instant, ZoneOffset.UTC);
    SimpleDateFormat dateFormat = new SimpleDateFormat(SimpleLogHandler.DEFAULT_TIMESTAMP_FORMAT);
    dateFormat.setTimeZone(TimeZone.getTimeZone(ZoneOffset.UTC));
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setClockForTesting(clock)
            .build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.

    assertThat(dateFormat.format(Date.from(instant))).isEqualTo("20150901-151754.");
    assertThat(handler.getCurrentLogFilePath().get().toString()).contains("20150901-151754.");
  }

  private static final class TrivialFormatter extends Formatter {
    @Override
    public String format(LogRecord rec) {
      return formatMessage(rec) + "\n";
    }
  }

  @Test
  public void testPublish() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setFormatter(new TrivialFormatter())
            .build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.
    Path logPath = handler.getCurrentLogFilePath().get();
    handler.close();

    assertThat(new String(Files.readAllBytes(logPath), UTF_8)).isEqualTo("Hello world\n");
  }

  @Test
  public void testSymbolicLinkDefaults() throws Exception {
    Path symlinkPath = Paths.get(tmp.getRoot().toString(), "hello");
    Files.createFile(symlinkPath);

    // On non-Windows platforms, expect to delete the file at symlinkPath and replace with a symlink
    // to the log.
    SimpleLogHandler handler = SimpleLogHandler.builder().setPrefix(symlinkPath.toString()).build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.

    if (OS.getCurrent() == OS.WINDOWS) {
      // On Windows, by default, only administrator accounts can create symbolic links.
      assertThat(handler.getSymbolicLinkPath()).isEmpty();
    } else {
      assertThat(handler.getSymbolicLinkPath()).isPresent();
      assertThat(handler.getSymbolicLinkPath().get().toString()).isEqualTo(symlinkPath.toString());
      assertThat(Files.isSymbolicLink(handler.getSymbolicLinkPath().get())).isTrue();
      assertThat(Files.readSymbolicLink(handler.getSymbolicLinkPath().get()).toString())
          .isEqualTo(handler.getCurrentLogFilePath().get().getFileName().toString());
    }
  }

  @Test
  public void testSymbolicLinkSetter() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setSymlinkName("bye")
            .build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.

    if (OS.getCurrent() == OS.WINDOWS) {
      // On Windows, by default, only administrator accounts can create symbolic links.
      assertThat(handler.getSymbolicLinkPath()).isEmpty();
    } else {
      assertThat(handler.getSymbolicLinkPath()).isPresent();
      assertThat(handler.getSymbolicLinkPath().get().toString())
          .isEqualTo(tmp.getRoot() + File.separator + "bye");
      assertThat(Files.isSymbolicLink(handler.getSymbolicLinkPath().get())).isTrue();
      assertThat(Files.readSymbolicLink(handler.getSymbolicLinkPath().get()).toString())
          .isEqualTo(handler.getCurrentLogFilePath().get().getFileName().toString());
    }
  }

  @Test
  public void testSymlinkEnabling() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setSymlinkName("bye")
            .setCreateSymlink(true)
            .build();
    assertThat(handler.getSymbolicLinkPath()).isPresent();
  }

  @Test
  public void testSymlinkDisabling() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setSymlinkName("bye")
            .setCreateSymlink(false)
            .build();
    assertThat(handler.getSymbolicLinkPath()).isEmpty();
  }

  @Test
  public void testSymbolicLinkInvalidPath() throws Exception {
    // "bye/bye" is invalid as a symlink path - it's not at the top level of log directory.
    SimpleLogHandler.Builder builder =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setSymlinkName("bye" + File.separator + "bye")
            .setCreateSymlink(true);
    assertThrows(IllegalArgumentException.class, () -> builder.build());
  }

  @Test
  public void testSymbolicLinkInitiallyInvalidReplaced() throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // On Windows, by default, only administrator accounts can create symbolic links.
      return;
    }
    Path symlinkPath = Paths.get(tmp.getRoot().toString(), "hello");
    Files.createSymbolicLink(symlinkPath, Paths.get("no-such-file"));

    // Expected to delete the (invalid) symlink and replace with a symlink to the log
    SimpleLogHandler handler =
        SimpleLogHandler.builder().setPrefix(symlinkPath.toString()).build();
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.

    assertThat(handler.getSymbolicLinkPath().get().toString()).isEqualTo(symlinkPath.toString());
    assertThat(Files.isSymbolicLink(handler.getSymbolicLinkPath().get())).isTrue();
    assertThat(Files.readSymbolicLink(handler.getSymbolicLinkPath().get()).toString())
        .isEqualTo(handler.getCurrentLogFilePath().get().getFileName().toString());
  }

  @Test
  public void testLogLevelEqualPublished() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "info")
            .setLogLevel(Level.INFO)
            .build();
    handler.publish(new LogRecord(Level.INFO, "Hello"));
    Optional<Path> logPath = handler.getCurrentLogFilePath();
    handler.close();

    assertThat(Files.size(logPath.get())).isGreaterThan(0L);
  }

  @Test
  public void testLogLevelHigherPublished() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "info")
            .setLogLevel(Level.INFO)
            .build();
    handler.publish(new LogRecord(Level.WARNING, "Hello"));
    Optional<Path> logPath = handler.getCurrentLogFilePath();
    handler.close();

    assertThat(Files.size(logPath.get())).isGreaterThan(0L);
  }

  @Test
  public void testLogLevelLowerNotPublished() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "info")
            .setLogLevel(Level.INFO)
            .build();
    handler.publish(new LogRecord(Level.FINE, "Hello"));
    Optional<Path> logPath = handler.getCurrentLogFilePath();
    handler.close();

    assertThat(logPath.isPresent()).isFalse();
  }

  @Test
  public void testLogLevelDefaultAllPublished() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder().setPrefix(tmp.getRoot() + File.separator + "all").build();
    handler.publish(new LogRecord(Level.FINEST, "Hello"));
    Optional<Path> logPath = handler.getCurrentLogFilePath();
    handler.close();

    assertThat(Files.size(logPath.get())).isGreaterThan(0L);
  }

  @Test
  public void testRotateLimitBytes() throws Exception {
    FakeClock clock = new FakeClock(Instant.parse("2018-01-01T12:00:00Z"), ZoneOffset.UTC);
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "limits")
            .setFormatter(new TrivialFormatter())
            .setRotateLimitBytes(16)
            .setClockForTesting(clock)
            .build();
    Optional<Path> symlinkPath = handler.getSymbolicLinkPath();
    handler.publish(new LogRecord(Level.SEVERE, "1234567" /* 8 bytes including "\n" */));
    Path firstLogPath = handler.getCurrentLogFilePath().get();
    clock.set(Instant.parse("2018-01-01T12:00:01Z")); // Ensure the next file has a different name.
    handler.publish(new LogRecord(Level.SEVERE, "1234567" /* 8 bytes including "\n" */));
    Path secondLogPath = handler.getCurrentLogFilePath().get();
    handler.publish(new LogRecord(Level.SEVERE, "1234567" /* 8 bytes including "\n" */));
    handler.close();

    if (symlinkPath.isPresent()) {
      // The symlink path is expected to be present on non-Windows platforms; see tests above.
      assertThat(Files.isSymbolicLink(symlinkPath.get())).isTrue();
      assertThat(Files.readSymbolicLink(symlinkPath.get()).toString())
          .isEqualTo(secondLogPath.getFileName().toString());
    }
    assertThat(Files.size(firstLogPath)).isEqualTo(16L /* including two "\n" */);
    assertThat(Files.size(secondLogPath)).isEqualTo(8L /* including "\n" */);
    try (DirectoryStream<Path> dirStream = Files.newDirectoryStream(tmp.getRoot().toPath())) {
      assertThat(dirStream).hasSize(3);
    }
  }

  private Path newFileWithContent(String name, String content) throws IOException {
    File file = tmp.newFile(name);
    try (OutputStreamWriter writer =
        new OutputStreamWriter(new FileOutputStream(file.getPath()), UTF_8)) {
      writer.write(content);
    }
    return file.toPath();
  }

  private Path newFileOfSize(String name, int size) throws IOException {
    char[] buf = new char[size];
    Arrays.fill(buf, '\n');
    return newFileWithContent(name, new String(buf));
  }

  @Test
  public void testOpenInAppendMode() throws Exception {
    Path logPath = newFileWithContent("hello.20150901-151754.log", "Previous logs\n");
    Instant instant = Instant.parse("2015-09-01T15:17:54Z");
    FakeClock clock = new FakeClock(instant, ZoneOffset.UTC);
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setPattern(".")
            .setExtension("log")
            .setFormatter(new TrivialFormatter())
            .setClockForTesting(clock)
            .build();
    handler.publish(new LogRecord(Level.SEVERE, "New logs"));
    assertThat(handler.getCurrentLogFilePath().get().toString()).isEqualTo(logPath.toString());
    handler.close();
    try (BufferedReader logReader =
        new BufferedReader(new InputStreamReader(new FileInputStream(logPath.toFile()), UTF_8))) {
      assertThat(logReader.readLine()).isEqualTo("Previous logs");
      assertThat(logReader.readLine()).isEqualTo("New logs");
    }
  }

  @Test
  public void testTotalLimit() throws Exception {
    String username = System.getProperty("user.name");
    if (Strings.isNullOrEmpty(username)) {
      username = "unknown_user";
    }
    String hostname = SimpleLogHandler.getLocalHostnameFirstComponent();
    String baseFilename = "hello." + hostname + "." + username + ".log.java.";
    Path nonLog = newFileOfSize("non_log", 16);
    Path missingDate = newFileOfSize(baseFilename + ".123", 16);
    Path invalidExtension = newFileOfSize(baseFilename + "19900101-120000.invalid", 16);
    Path oldDeleted1 = newFileOfSize(baseFilename + "19900101-120000.123", 16);
    Path oldDeleted2 = newFileOfSize(baseFilename + "19950101-120000.123", 16);
    Path keptThenDeleted = newFileOfSize(baseFilename + "19990101-120000.123", 16);
    Path kept = newFileOfSize(baseFilename + "19990606-060000.123", 16);

    FakeClock clock = new FakeClock(Instant.parse("2018-01-01T12:00:00Z"), ZoneOffset.UTC);
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setPattern(".%h.%u.log.java.")
            .setFormatter(new TrivialFormatter())
            .setRotateLimitBytes(16)
            .setTotalLimitBytes(40)
            .setClockForTesting(clock)
            .build();
    // Print 8 bytes into the log file. Opening the log file triggers deletion of old logs.
    handler.publish(new LogRecord(Level.SEVERE, "1234567" /* 8 bytes including "\n" */));

    // We expect handler to delete all but 32 = 40 - 8 bytes worth of old log files.
    assertThat(Files.exists(nonLog)).isTrue();
    assertThat(Files.exists(missingDate)).isTrue();
    assertThat(Files.exists(invalidExtension)).isTrue();
    assertThat(Files.exists(oldDeleted1)).isFalse();
    assertThat(Files.exists(oldDeleted2)).isFalse();
    assertThat(Files.exists(keptThenDeleted)).isTrue();
    assertThat(Files.exists(kept)).isTrue();

    handler.publish(new LogRecord(Level.SEVERE, "1234567" /* 8 bytes including "\n" */));
    Path currentLogPath = handler.getCurrentLogFilePath().get();
    handler.close();

    // We expect another old log file to be deleted after rotation.
    assertThat(Files.exists(keptThenDeleted)).isFalse();
    assertThat(Files.exists(kept)).isTrue();
    assertThat(Files.exists(currentLogPath)).isTrue();
  }

  @Test
  public void getLoggerFilePath_onSimpleLogHandler_withFile_returnsPath() throws Exception {
    HandlerQuerier handlerQuerier = new HandlerQuerier();
    SimpleLogHandler handler =
        SimpleLogHandler.builder().setPrefix(tmp.getRoot() + File.separator + "hello").build();
    Logger logger = Logger.getAnonymousLogger();
    logger.addHandler(handler);
    handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // Ensure log file is opened.

    Optional<Path> retrievedLogPath = handlerQuerier.getLoggerFilePath(logger);

    assertThat(retrievedLogPath).isPresent();
    assertThat(retrievedLogPath.get().toString())
        .startsWith(tmp.getRoot() + File.separator + "hello");

    handler.close();
  }

  @Test
  public void getLoggerFilePath_onSimpleLogHandler_withoutFile_returnsEmpty() throws Exception {
    HandlerQuerier handlerQuerier = new HandlerQuerier();
    SimpleLogHandler handler =
        SimpleLogHandler.builder().setPrefix(tmp.getRoot() + File.separator + "hello").build();
    Logger logger = Logger.getAnonymousLogger();
    logger.addHandler(handler);

    assertThat(handlerQuerier.getLoggerFilePath(logger)).isEmpty();
  }

  @Test
  public void getLoggerFilePath_onUnsupportedLogHandler_fails() throws Exception {
    HandlerQuerier handlerQuerier = new HandlerQuerier();
    FileHandler unsupportedHandler = new FileHandler(tmp.getRoot() + File.separator + "hello");
    Logger logger = Logger.getAnonymousLogger();
    logger.addHandler(unsupportedHandler);

    assertThrows(IOException.class, () -> handlerQuerier.getLoggerFilePath(logger));

    unsupportedHandler.close();
  }

  @Test
  public void getLoggerFilePath_onMissingLogHandler_fails() throws Exception {
    HandlerQuerier handlerQuerier = new HandlerQuerier();
    Logger logger = Logger.getAnonymousLogger();

    assertThrows(IOException.class, () -> handlerQuerier.getLoggerFilePath(logger));
  }

  @Test
  public void publish_handlesInterrupt() throws Exception {
    SimpleLogHandler handler =
        SimpleLogHandler.builder()
            .setPrefix(tmp.getRoot() + File.separator + "hello")
            .setFormatter(new TrivialFormatter())
            .build();
    Thread t =
        new Thread(
            () -> {
              Thread.currentThread().interrupt();
              handler.publish(new LogRecord(Level.SEVERE, "Hello world")); // To open the log file.
              assertThat(Thread.currentThread().isInterrupted()).isTrue();
              handler.flush();
              assertThat(Thread.currentThread().isInterrupted()).isTrue();
              handler.close();
              assertThat(Thread.currentThread().isInterrupted()).isTrue();
            });
    t.run();
    t.join();
  }
}
