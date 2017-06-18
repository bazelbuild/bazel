package joptsimple.examples;

import static java.util.Arrays.*;
import static java.util.Collections.*;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class ShortOptionsWithArgumentsTest {
    @Test
    public void allowsOptionsToAcceptArguments() {
        OptionParser parser = new OptionParser( "fc:q::" );

        OptionSet options = parser.parse( "-f", "-c", "foo", "-q" );

        assertTrue( options.has( "f" ) );

        assertTrue( options.has( "c" ) );
        assertTrue( options.hasArgument( "c" ) );
        assertEquals( "foo", options.valueOf( "c" ) );
        assertEquals( asList( "foo" ), options.valuesOf( "c" ) );

        assertTrue( options.has( "q" ) );
        assertFalse( options.hasArgument( "q" ) );
        assertNull( options.valueOf( "q" ) );
        assertEquals( emptyList(), options.valuesOf( "q" ) );
    }
}
