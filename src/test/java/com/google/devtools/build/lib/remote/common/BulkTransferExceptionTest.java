package com.google.devtools.build.lib.remote.common;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInputDepOwnerMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BulkTransferExceptionTest {
  private Scratch scratch;
  private Path execDir;

  @Before
  public void before() throws Exception {
    scratch = new Scratch();
    execDir = scratch.dir("/base/exec");
  }

  @Test
  public void asLostInputsEmpty() {
    LostInputsExecException e = new BulkTransferException().asLostInputsExecException();
    assertThat(e.getLostInputs()).isEmpty();
  }

  @Test
  public void asLostInputSingle() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    LostInputsExecException fooException = exception("foo");
    bulkTransferException.addSuppressed(new IOException(fooException));
    LostInputsExecException e = bulkTransferException.asLostInputsExecException();
    assertThat(e.getLostInputs()).containsKey("foo-digest");
    assertThat(e.getOwners().getDepOwners(artifact("foo"))).contains(artifact("foo"));
  }

  @Test
  public void asLostInputsMultiple() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    LostInputsExecException fooException = exception("foo");
    LostInputsExecException barException = exception("bar");
    bulkTransferException.addSuppressed(new IOException(fooException));
    bulkTransferException.addSuppressed(new IOException(barException));
    LostInputsExecException e = bulkTransferException.asLostInputsExecException();

    assertThat(e.getLostInputs()).containsKey("foo-digest");
    assertThat(e.getOwners().getDepOwners(artifact("foo"))).contains(artifact("foo"));

    assertThat(e.getLostInputs()).containsKey("bar-digest");
    assertThat(e.getOwners().getDepOwners(artifact("bar"))).contains(artifact("bar"));
  }

  @Test
  public void asLostInputsWrongType() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.addSuppressed(new IOException(new RuntimeException("blah")));
    LostInputsExecException e = bulkTransferException.asLostInputsExecException();
    assertThat(e).isNull();
  }

  private Artifact artifact(String name) {
    return ActionsTestUtil.createArtifact(ArtifactRoot.asDerivedRoot(execDir, RootType.Output, "root"), name);
  }

  private LostInputsExecException exception(String name) {
    Artifact artifact = artifact(name);
    ActionInputDepOwnerMap owners = new ActionInputDepOwnerMap(ImmutableSet.of(artifact));
    owners.addOwner(artifact, artifact);
    return new LostInputsExecException(ImmutableMap.of(name + "-digest", artifact), owners);
  }
}
