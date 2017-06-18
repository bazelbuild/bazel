package joptsimple.examples;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class ShortOptionsWithArgumentPositioningTest {
    @Test
    public void allowsDifferentFormsOfPairingArgumentWithOption() {
        OptionParser parser = new OptionParser( "a:b:c::" );

        OptionSet options = parser.parse( "-a", "foo", "-bbar", "-c=baz" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.hasArgument( "a" ) );
        assertEquals( "foo", options.valueOf( "a" ) );

        assertTrue( options.has( "b" ) );
        assertTrue( options.hasArgument( "b" ) );
        assertEquals( "bar", options.valueOf( "b" ) );

        assertTrue( options.has( "c" ) );
        assertTrue( options.hasArgument( "c" ) );
        assertEquals( "baz", options.valueOf( "c" ) );
    }
}
