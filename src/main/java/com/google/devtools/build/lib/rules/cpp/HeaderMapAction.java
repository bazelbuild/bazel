package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.io.OutputStream;//?
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.util.Map;

@Immutable
public final class HeaderMapAction extends AbstractFileWriteAction {

  private static final String GUID = "4f407081-1951-40c1-befc-d6b4daff5de3";

  // C++ header map of the current target
  private final ImmutableMap <String, String> headerMap;

  public HeaderMapAction(
      ActionOwner owner,
      ImmutableMap <String, String> headerMap,
      Artifact output
      ) {
    super(
        owner,
        ImmutableList.of(),
        output,
        /* makeExecutable= */ false);
    this.headerMap = headerMap;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext context)  {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        ClangHeaderMap headerMap1 = new ClangHeaderMap(headerMap);
        WritableByteChannel channel = Channels.newChannel(out);
        headerMap1.writeToChannel(channel);
        out.flush();
        out.close();
      }
    };
  }

  @Override
  public String getMnemonic() {
    return "CppHeaderMap";
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint f) {
    f.addString(GUID);
    for(Map.Entry<String, String> entry: headerMap.entrySet()){
      String key = entry.getKey();
      String path = entry.getValue();
      f.addString(key + path);
    }
  }
}
