package com.google.devtools.build.lib.rules.cpp;

//TODO: Cleanup Imports
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;//?
import java.io.DataOutputStream;//?
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.nio.channels.FileChannel;
import java.io.FileOutputStream;

@Immutable
public final class HeaderMapAction extends AbstractFileWriteAction {

  private static final String GUID = "4f407081-1951-40c1-befc-d6b4daff5de3";

  // C++ header map of the current target
  private final Map <String, String> headerMap;

  public HeaderMapAction(
      ActionOwner owner,
      Map <String, String> headerMap,
      Artifact output
      ) {
    super(
        owner,
        ImmutableList.<Artifact>builder().build(),
        output,
        /*makeExecutable=*/ false);
    this.headerMap = headerMap;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)  {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        ClangHeaderMap hmap = new ClangHeaderMap(headerMap);
        ByteBuffer b = hmap.buff;
        b.flip();
        WritableByteChannel channel = Channels.newChannel(out);
        channel.write(b);
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
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint f)
      throws CommandLineExpansionException {
    f.addString(GUID);
    for(Map.Entry<String, String> entry: headerMap.entrySet()){
      String key = entry.getKey();
      String path = entry.getValue();
      f.addString(key + path);
    }
  }
}
