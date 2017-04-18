package joptsimple.examples;

import static java.util.Arrays.*;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class SignallingEndOfOptionsTest {
    @Test
    public void doubleHyphenSignalsEndOfOptions() {
        OptionParser parser = new OptionParser( "ab:c::de:f::" );

        OptionSet options = parser.parse( "-a", "-b=foo", "-c=bar", "--", "-d", "-e", "baz", "-f", "biz" );

        assertTrue( options.has( "a" ) );
        assertFalse( options.hasArgument( "a" ) );
        assertTrue( options.has( "b" ) );
        assertTrue( options.hasArgument( "b" ) );
        assertEquals( asList( "foo" ), options.valuesOf( "b" ) );
        assertTrue( options.has( "c" ) );
        assertTrue( options.hasArgument( "c" ) );
        assertEquals( asList( "bar" ), options.valuesOf( "c" ) );
        assertFalse( options.has( "d" ) );
        assertFalse( options.has( "e" ) );
        assertFalse( options.has( "f" ) );
        assertEquals( asList( "-d", "-e", "baz", "-f", "biz" ), options.nonOptionArguments() );
    }
}
