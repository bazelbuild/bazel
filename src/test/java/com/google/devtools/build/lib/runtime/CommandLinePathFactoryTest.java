package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandLinePathFactory}. */
@RunWith(JUnit4.class)
public class CommandLinePathFactoryTest {
  @Test
  public void createFromAbsolutePath() {
    CommandLinePathFactory factory = new CommandLinePathFactory(ImmutableMap.of());

    assertThat(factory.create("/absolute/path/1"))
        .isEqualTo(PathFragment.create("/absolute/path/1"));
    assertThat(factory.create("/absolute/path/2"))
        .isEqualTo(PathFragment.create("/absolute/path/2"));
  }

  @Test
  public void createWithNamedRoot() {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(
            ImmutableMap.<String, PathFragment>builder()
                .put("workspace", PathFragment.create("/path/to/workspace"))
                .put("output_base", PathFragment.create("/path/to/output/base"))
                .build());

    assertThat(factory.create("/absolute/path/1"))
        .isEqualTo(PathFragment.create("/absolute/path/1"));
    assertThat(factory.create("/absolute/path/2"))
        .isEqualTo(PathFragment.create("/absolute/path/2"));

    assertThat(factory.create("%workspace%/foo"))
        .isEqualTo(PathFragment.create("/path/to/workspace/foo"));
    assertThat(factory.create("%workspace%/foo/bar"))
        .isEqualTo(PathFragment.create("/path/to/workspace/foo/bar"));

    assertThat(factory.create("%output_base%/foo"))
        .isEqualTo(PathFragment.create("/path/to/output/base/foo"));
    assertThat(factory.create("%output_base%/foo/bar"))
        .isEqualTo(PathFragment.create("/path/to/output/base/foo/bar"));
  }

  @Test
  public void UnknownRoot() {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(ImmutableMap.of("a", PathFragment.create("/path/to/a")));

    assertThrows(IllegalArgumentException.class, () -> factory.create("%workspace%/foo"));
    assertThrows(IllegalArgumentException.class, () -> factory.create("%output_base%/foo"));
  }
}
