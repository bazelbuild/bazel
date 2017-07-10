package joptsimple.examples;

import static java.util.Arrays.*;
import static java.util.Collections.*;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class PosixlyCorrectTest {
    @Test
    public void supportsPosixlyCorrectBehavior() {
        OptionParser parser = new OptionParser( "i:j::k" );
        String[] arguments = { "-ibar", "-i", "junk", "xyz", "-jixnay", "foo", "-k", "blah", "--", "bah" };

        OptionSet options = parser.parse( arguments );

        assertTrue( options.has( "i" ) );
        assertTrue( options.has( "j" ) );
        assertTrue( options.has( "k" ) );
        assertEquals( asList( "bar", "junk" ), options.valuesOf( "i" ) );
        assertEquals( asList( "ixnay" ), options.valuesOf( "j" ) );
        assertEquals( asList( "xyz", "foo", "blah", "bah" ), options.nonOptionArguments() );

        parser.posixlyCorrect( true );
        options = parser.parse( arguments );

        assertTrue( options.has( "i" ) );
        assertFalse( options.has( "j" ) );
        assertFalse( options.has( "k" ) );
        assertEquals( asList( "bar", "junk" ), options.valuesOf( "i" ) );
        assertEquals( emptyList(), options.valuesOf( "j" ) );
        assertEquals( asList( "xyz", "-jixnay", "foo", "-k", "blah", "--", "bah" ), options.nonOptionArguments() );
    }
}
