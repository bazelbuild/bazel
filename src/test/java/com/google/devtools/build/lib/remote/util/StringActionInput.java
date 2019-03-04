package com.google.devtools.build.lib.remote.util;

import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/** A virtual action input backed by a string */
public final class StringActionInput implements VirtualActionInput {
  private final String contents;
  private final PathFragment execPath;

  public StringActionInput(String contents, PathFragment execPath) {
    this.contents = contents;
    this.execPath = execPath;
  }

  @Override
  public void writeTo(OutputStream out) throws IOException {
    out.write(contents.getBytes(StandardCharsets.UTF_8));
  }

  @Override
  public ByteString getBytes() throws IOException {
    ByteString.Output out = ByteString.newOutput();
    writeTo(out);
    return out.toByteString();
  }

  @Override
  public String getExecPathString() {
    return execPath.getPathString();
  }

  @Override
  public PathFragment getExecPath() {
    return execPath;
  }
}
