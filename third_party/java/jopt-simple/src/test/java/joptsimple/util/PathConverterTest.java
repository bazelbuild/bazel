package joptsimple.util;

import java.nio.file.Files;
import java.nio.file.Path;

import joptsimple.ValueConversionException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static joptsimple.util.PathProperties.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class PathConverterTest {
    @Rule
    public ExpectedException exception = ExpectedException.none();

    @Test
    public void convertsValuesToPaths() throws Exception {
        Path path = Files.createTempFile( "prefix", null );
        path.toFile().deleteOnExit();

        String pathName = path.toString();

        assertEquals( path, new PathConverter( null ).convert( pathName ) );
        assertEquals( path, new PathConverter().convert( pathName ) );
    }

    @Test
    public void answersCorrectValueType() {
        assertEquals( Path.class, new PathConverter().valueType() );
    }

    @Test
    public void testReadableAndOverwritableFile() throws Exception {
        Path path = Files.createTempFile( "prefix", null );
        path.toFile().deleteOnExit();

        String pathName = path.toString();

        assertTrue( Files.isReadable( new PathConverter( READABLE ).convert( pathName ) ) );
        assertTrue( Files.exists( new PathConverter( READABLE ).convert( pathName ) ) );
        assertTrue( Files.isWritable( new PathConverter( READABLE ).convert( pathName ) ) );
        assertTrue( Files.isWritable( new PathConverter( FILE_OVERWRITABLE).convert( pathName ) ) );
    }

    @Test
    public void testNotExisting() throws Exception {
        Path path = Files.createTempFile( "prefix", null );

        Files.deleteIfExists( path );

        assertFalse( Files.exists( new PathConverter( NOT_EXISTING ).convert( path.toString() ) ) );
    }

    @Test
    public void testNotReadable() throws Exception {
        Path path = Files.createTempFile( "prefix", null );
        String pathName = path.toString();
        Files.deleteIfExists( path );

        exception.expect( ValueConversionException.class );
        exception.expectMessage( "File [" + pathName );

        new PathConverter( READABLE ).convert( pathName );
    }

    @Test
    public void testDirectoryExisting() throws Exception {
        Path path = Files.createTempDirectory( "prefix" );
        path.toFile().deleteOnExit();

        String pathName = path.toString();

        assertTrue( Files.isDirectory( new PathConverter( DIRECTORY_EXISTING ).convert( pathName ) ) );
    }

    @Test
    public void testDirectoryNotOverwritable() throws Exception {
        Path path = Files.createTempDirectory( "prefix" );
        path.toFile().deleteOnExit();

        String pathName = path.toString();

        exception.expect( ValueConversionException.class );
        exception.expectMessage( "File [" + pathName );

        new PathConverter( FILE_OVERWRITABLE).convert( pathName );
    }

    @Test
    public void testNotExistingNotOverwritable() throws Exception {
        Path path = Files.createTempDirectory( "prefix" );
        String pathName = path.toString();
        Files.deleteIfExists( path );

        exception.expect( ValueConversionException.class );
        exception.expectMessage( "File [" + pathName );

        new PathConverter( FILE_OVERWRITABLE ).convert( pathName );
    }
}
