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
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.when;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.logging.FileHandler;
import java.util.logging.LogManager;
import java.util.logging.Logger;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for the {@link FileHandlerQuerier} class. */
@RunWith(JUnit4.class)
public class FileHandlerQuerierTest {
  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  private Logger getLoggerWithFileHandler(FileHandler handler) {
    Logger logger = Logger.getAnonymousLogger();
    logger.addHandler(handler);
    return logger;
  }

  private Logger getLoggerWithFileHandler(Path logPath) throws IOException {
    return getLoggerWithFileHandler(new FileHandler(logPath.toString()));
  }

  @Test
  public void getLoggerFilePath_onExpectedConfigurationOpenFile_returnsPath() throws Exception {
    Path configuredLogPath = Paths.get(tmp.getRoot().toString(), "hello.log");
    LogManager mockLogManager = Mockito.mock(LogManager.class);
    when(mockLogManager.getProperty("java.util.logging.FileHandler.pattern"))
        .thenReturn(configuredLogPath.toString());
    Logger logger = getLoggerWithFileHandler(configuredLogPath);
    FileHandlerQuerier handlerQuerier = new FileHandlerQuerier(() -> mockLogManager);

    Optional<Path> retrievedLogPath = handlerQuerier.getLoggerFilePath(logger);

    assertThat(retrievedLogPath).isPresent();
    assertThat(retrievedLogPath.get().toString()).isEqualTo(configuredLogPath.toString());
  }

  @Test
  public void getLoggerFilePath_onExpectedConfigurationClosedFile_returnsEmpty() throws Exception {
    Path configuredLogPath = Paths.get(tmp.getRoot().toString(), "hello.log");
    LogManager mockLogManager = Mockito.mock(LogManager.class);
    when(mockLogManager.getProperty("java.util.logging.FileHandler.pattern"))
        .thenReturn(configuredLogPath.toString());
    FileHandler handler = new FileHandler(configuredLogPath.toString());
    Logger logger = getLoggerWithFileHandler(handler);
    FileHandlerQuerier handlerQuerier = new FileHandlerQuerier(() -> mockLogManager);
    handler.close();

    assertThat(handlerQuerier.getLoggerFilePath(logger)).isEmpty();
  }

  @Test
  public void getLoggerFilePath_onMissingConfiguration_fails() throws Exception {
    Path configuredLogPath = Paths.get(tmp.getRoot().toString(), "hello.log");
    LogManager mockLogManager = Mockito.mock(LogManager.class);
    when(mockLogManager.getProperty("java.util.logging.FileHandler.pattern")).thenReturn(null);
    Logger logger = getLoggerWithFileHandler(configuredLogPath);
    FileHandlerQuerier handlerQuerier = new FileHandlerQuerier(() -> mockLogManager);

    assertThrows(IllegalStateException.class, () -> handlerQuerier.getLoggerFilePath(logger));
  }

  @Test
  public void getLoggerFilePath_onVariablesInPath_fails() throws Exception {
    LogManager mockLogManager = Mockito.mock(LogManager.class);
    when(mockLogManager.getProperty("java.util.logging.FileHandler.pattern"))
        .thenReturn(tmp.getRoot() + File.separator + "hello_%u.log");
    Logger logger =
        getLoggerWithFileHandler(Paths.get(tmp.getRoot().toString(), "hello_0.log"));
    FileHandlerQuerier handlerQuerier = new FileHandlerQuerier();

    assertThrows(IllegalStateException.class, () -> handlerQuerier.getLoggerFilePath(logger));
  }

  @Test
  public void getLoggerFilePath_onUnsupportedLogHandler_fails() throws Exception {
    FileHandlerQuerier handlerQuerier = new FileHandlerQuerier();
    Logger logger = Logger.getAnonymousLogger();
    logger.addHandler(
        SimpleLogHandler.builder().setPrefix(tmp.getRoot() + File.separator + "hello.log").build());

    assertThrows(IOException.class, () -> handlerQuerier.getLoggerFilePath(logger));
  }

  @Test
  public void getLoggerFilePath_onMissingLogHandler_fails() throws Exception {
    FileHandlerQuerier handlerQuerier = new FileHandlerQuerier();
    Logger logger = Logger.getAnonymousLogger();

    assertThrows(IOException.class, () -> handlerQuerier.getLoggerFilePath(logger));
  }
}
