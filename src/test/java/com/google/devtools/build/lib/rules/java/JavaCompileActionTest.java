package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.*;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
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

@RunWith(JUnit4.class)
public class JavaCompileActionTest {
    private static final Label NULL_LABEL = Label.parseAbsoluteUnchecked("//null/action:owner");

    @Mock
    private BuildConfiguration configuration;

    @Before
    public final void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    private InMemoryFileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    private Path workingDirectory = fs.getPath("/etc/something");

    // The goal of this test is to FAIL in case any new field was added.
    @Test
    public void test_JavaCompileAction() {
        JavaCompileAction action = getDefaultJavaCompileAction();
        ActionAnalysisMetadata modified = action.addExecutionInfo(ImmutableMap.of("one", "two"));

        // check that returned action is of the same type
        assertThat(modified).isInstanceOf(action.getClass());

        JavaCompileAction modifiedAction = (JavaCompileAction) modified;
        // check that all the class fields are the same in both instances

        Field[] fields = action.getClass().getFields();
        for (Field f : fields) {
            try {
                assertThat(f.equals(modifiedAction.getClass().getField(f.getName()))).isTrue();
            } catch (NoSuchFieldException e) {
                Assert.fail(String.format("It seems that you've added a new field %s to the action %s. " +
                        "This field is not set properly in the constructor used for addExecutionInfo method. " +
                        "Please add it there", f.getName(), action.getClass()));
            }
        }
    }

    private JavaCompileAction getDefaultJavaCompileAction() {
        return new JavaCompileAction(
                    JavaCompileAction.CompilationType.JAVAC,
                    ActionOwner.SYSTEM_ACTION_OWNER,
                    ActionEnvironment.EMPTY,
                    NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
                    EmptyRunfilesSupplier.INSTANCE,
                    new LazyString() {
                        @Override
                        public String toString() {
                            return "mu";
                        }
                    },
                    NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
                    NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
                    null,
                    new NestedSetBuilder<Artifact>(Order.COMPILE_ORDER).add(new Artifact.SourceArtifact(ArtifactRoot.asSourceRoot(Root.fromPath(workingDirectory)), PathFragment.create("/some"), new NullArtifactOwner())).build(),
                    ImmutableMap.of("some", "thing"),
                    null,
                    null,
                    null,
                    configuration,
                    null,
                    null,
                    null);
    }

    static class NullArtifactOwner implements ArtifactOwner {
        private NullArtifactOwner() {
        }

        @Override
        public Label getLabel() {
            return NULL_LABEL;
        }
    }
}