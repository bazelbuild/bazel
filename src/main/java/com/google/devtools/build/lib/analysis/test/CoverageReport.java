package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;

/** This event is used to notify about a successfully generated coverage report. */
public final class CoverageReport implements ExtendedEventHandler.Postable {
  private final ImmutableList<Path> files;

  public CoverageReport(
      ImmutableList<Path> files) {
    this.files = files;
  }

  public ImmutableList<Path> getFiles() {
    return files;
  }
}
