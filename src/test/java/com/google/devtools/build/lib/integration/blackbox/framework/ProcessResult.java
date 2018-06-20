package com.google.devtools.build.lib.integration.blackbox.framework;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.util.StringUtilities;
import java.util.List;

@AutoValue
public abstract class ProcessResult {

  static ProcessResult create(int exitCode, List<String> out, List<String> err) {
    return new AutoValue_ProcessResult(exitCode, out, err);
  }

  abstract int exitCode();

  abstract List<String> out();

  abstract List<String> err();

  public String outString() {
    return StringUtilities.joinLines(out());
  }

  public String errString() {
    return StringUtilities.joinLines(err());
  }
}
