package com.google.devtools.build.lib.buildeventstream;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.NullUploader;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)

/** Tests for {@link NullUploader}. */
public class NullUploaderTest {

  @Test
  public void testPathToUriString() {
    // See https://blogs.msdn.microsoft.com/ie/2006/12/06/file-uris-in-windows/
    assertThat(BuildEventArtifactUploader.NullUploader.pathToUriString("C:/Temp/Foo Bar.txt"))
        .isEqualTo("file:///C:/Temp/Foo%20Bar.txt");
  }

}
