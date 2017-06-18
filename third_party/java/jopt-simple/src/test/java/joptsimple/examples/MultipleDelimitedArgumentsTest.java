package joptsimple.examples;

import java.io.File;

import static java.io.File.*;
import static java.util.Arrays.*;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import joptsimple.OptionSpec;
import org.junit.Test;

import static joptsimple.examples.Strings.*;
import static org.junit.Assert.*;

public class MultipleDelimitedArgumentsTest {
    @Test
    public void supportsMultipleDelimitedArguments() {
        OptionParser parser = new OptionParser();
        OptionSpec<File> path = parser.accepts( "path" ).withRequiredArg().ofType( File.class )
            .withValuesSeparatedBy( pathSeparatorChar );

        OptionSet options = parser.parse( "--path", join( pathSeparatorChar, "/tmp", "/var", "/opt" ) );

        assertTrue( options.has( path ) );
        assertTrue( options.hasArgument( path ) );
        assertEquals( asList( new File( "/tmp" ), new File( "/var" ), new File( "/opt" ) ), options.valuesOf( path ) );
    }
}
