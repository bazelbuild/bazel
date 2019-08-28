package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.lang.reflect.Field;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.*;

@RunWith(JUnit4.class)
public class CppCompileActionTest extends BuildViewTestCase {

    @Mock
    private BuildConfiguration configuration;

    @Before
    public final void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    private InMemoryFileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    private Path workingDirectory = fs.getPath("/etc/something");

    // TODO(ishikhman): move to shared place? can I?
    // The goal of this test is to FAIL in case any new field was added.
    @Test
    public void test_JavaCompileAction() throws Exception {
        CppCompileAction action = getDefaultAction();
        ActionAnalysisMetadata modified = action.addExecutionInfo(ImmutableMap.of("one", "two"));

        // check that returned action is of the same type
        assertThat(modified).isInstanceOf(action.getClass());

        CppCompileAction modifiedAction = (CppCompileAction) modified;
        // check that all the class fields are the same in both instances

        try {
            assertTrue(action.deepEquals(modifiedAction));
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (NoSuchFieldException e) {
            Assert.fail(e.getLocalizedMessage());
        }
    }

    private CppCompileAction getDefaultAction() throws Exception {
        ConfiguredTarget cclibrary =
                ScratchAttributeWriter.fromLabelString(this, "cc_library", "//cclib")
                        .setList("srcs", "a.cc")
                        .setList("copts", "foobar-$(ABI)")
                        .write();
        CppCompileAction compileAction =
                (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/cclib/a.o", cclibrary));
        return compileAction;
    }

}