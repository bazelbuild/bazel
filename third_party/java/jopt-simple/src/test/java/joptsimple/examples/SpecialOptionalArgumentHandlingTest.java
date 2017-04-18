package joptsimple.examples;

import static java.util.Arrays.*;
import static java.util.Collections.*;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class SpecialOptionalArgumentHandlingTest {
    @Test
    public void handlesNegativeNumberOptionalArguments() {
        OptionParser parser = new OptionParser();
        parser.accepts( "a" ).withOptionalArg().ofType( Integer.class );
        parser.accepts( "2" );

        OptionSet options = parser.parse( "-a", "-2" );

        assertTrue( options.has( "a" ) );
        assertFalse( options.has( "2" ) );
        assertEquals( asList( -2 ), options.valuesOf( "a" ) );

        options = parser.parse( "-2", "-a" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.has( "2" ) );
        assertEquals( emptyList(), options.valuesOf( "a" ) );
    }
}
